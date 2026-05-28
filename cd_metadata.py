#!/usr/bin/env python3
"""Generate CD metadata from a webpage URL using LM Studio.

Workflow: you've ripped a CD that MusicBrainz/Discogs don't have. Point this
at a webpage that lists the release (label site, Bandcamp, artist page,
Wikipedia, blog review) and a locally-running LM Studio model extracts
structured metadata aligned to your actual ripped tracks.

Requirements:
    - LM Studio app running with a model loaded (https://lmstudio.ai)
    - uv (https://docs.astral.sh/uv/) — `uv sync` installs deps into .venv

Usage (from the project directory):
    uv run cd_metadata.py --url URL --dir PATH_TO_RIPS [--write] [--rename]
    uv run cd_metadata.py --url URL --dir PATH_TO_RIPS --out tags.json

With --rename, the directory is renamed to "%A - %T (%y) [%f]" and each
file to "%n. %t" preserving its original extension.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import lmstudio as lms
import mutagen
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

AUDIO_EXTS = {".flac", ".wav", ".mp3",
              ".m4a", ".aiff", ".aif", ".ogg", ".opus"}
PAGE_CHAR_BUDGET = 60_000
INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Wikipedia/CMS disambiguator suffixes that pages append to titles but should
# never be tagged onto an album or track (e.g., "Discovery (album)").
WIKI_DISAMBIGUATORS = {
    "album", "ep", "song", "single", "track", "soundtrack", "ost",
    "film", "movie", "band", "artist", "musician", "group",
    "text", "article", "page", "disambiguation",
    "compilation", "live album", "studio album", "mixtape",
}
TRAILING_PAREN_RE = re.compile(r"\s*\(([^()]+)\)\s*$")

# Discogs blocks scraping; use their public JSON API instead.
DISCOGS_URL_RE = re.compile(
    r"discogs\.com/(?:[a-z]{2}/)?(release|master|r|m)/(\d+)", re.IGNORECASE
)
DISCOGS_API = "https://api.discogs.com"
DISCOGS_KEEP_KEYS = {
    "title", "artists", "artists_sort", "year", "released", "released_formatted",
    "labels", "companies", "formats", "genres", "styles", "country",
    "tracklist", "identifiers", "extraartists", "notes",
}
USER_AGENT = "cd-metadata/0.1 (+https://github.com/local/cd-metadata)"


class Track(BaseModel):
    track_number: int
    disc_number: int = 1
    title: str
    artist: str | None = None
    composer: str | None = None
    duration_seconds: float | None = None


class AlbumMetadata(BaseModel):
    album: str
    album_artist: str
    date: str | None = None
    label: str | None = None
    catalog_number: str | None = None
    genre: str | None = None
    tracks: list[Track]
    notes: str | None = None


SYSTEM_PROMPT = """You extract album metadata from a webpage and align it to a user's ripped CD tracks.

Rules:
- Use ONLY information present on the page. Never invent values.
- For any field not stated on the page, return null.
- The user provides their actual track count and per-track durations in file order. Align the page's tracklist to those tracks IN ORDER.
- If the page's tracklist count does not match the rip's track count, fill what you can and explain the discrepancy in `notes`.
- If the page lists durations, verify alignment; flag mismatches greater than 5 seconds in `notes`.
- Emit track_number starting at 1, matching the order of the user's files.
- Dates: "YYYY" or "YYYY-MM-DD" only if the page is that specific.
- If the page contains JSON-LD structured data, prefer it over prose.
- Strip Wikipedia/CMS disambiguator suffixes from `album` and track titles. These look like a single short descriptor in trailing parentheses — "(album)", "(EP)", "(song)", "(soundtrack)", "(text)", "(disambiguation)", "(film)", "(band)", etc. They are page-routing artifacts, not part of the real title. Preserve legitimate parentheticals that are part of the title itself (e.g., "(What's the Story) Morning Glory?", "Smells Like Teen Spirit (Single Edit)").
"""


def load_local_tracks(audio_dir: Path) -> list[dict]:
    files = sorted(p for p in audio_dir.iterdir()
                   if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        sys.exit(f"No audio files found in {audio_dir}")
    rows = []
    for idx, f in enumerate(files, start=1):
        m = mutagen.File(f)
        dur = round(m.info.length, 2) if m and m.info else None
        rows.append({"position": idx, "filename": f.name,
                    "duration_seconds": dur})
    return rows


def fetch_discogs(kind: str, ident: str) -> str:
    endpoint = "masters" if kind.lower() in ("master", "m") else "releases"
    api_url = f"{DISCOGS_API}/{endpoint}/{ident}"
    resp = requests.get(
        api_url,
        timeout=30,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    trimmed = {k: v for k, v in data.items() if k in DISCOGS_KEEP_KEYS}
    return (
        f"=== DISCOGS API ({endpoint}/{ident}) ===\n"
        + json.dumps(trimmed, indent=2, ensure_ascii=False)
    )


def fetch_page_text(url: str) -> str:
    m = DISCOGS_URL_RE.search(url)
    if m:
        return fetch_discogs(m.group(1), m.group(2))

    resp = requests.get(
        url, timeout=30, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    raw = resp.text

    ld_blocks: list[str] = []
    for s in BeautifulSoup(raw, "html.parser").find_all(
        "script", type="application/ld+json"
    ):
        if s.string:
            ld_blocks.append(s.string.strip())

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    body_text = soup.get_text("\n", strip=True)

    parts = []
    if ld_blocks:
        parts.append("=== JSON-LD ===\n" + "\n---\n".join(ld_blocks))
    parts.append("=== PAGE TEXT ===\n" + body_text)
    return "\n\n".join(parts)


def build_user_message(url: str, page_text: str, local_tracks: list[dict]) -> str:
    track_list = "\n".join(
        f"  {t['position']}. {t['filename']} ({t['duration_seconds']}s)"
        for t in local_tracks
    )
    truncated = page_text[:PAGE_CHAR_BUDGET]
    if len(page_text) > PAGE_CHAR_BUDGET:
        truncated += f"\n[... {len(page_text) - PAGE_CHAR_BUDGET} more chars truncated]"
    return (
        f"Source URL: {url}\n\n"
        f"My ripped CD has {len(local_tracks)} track(s) in this order:\n{track_list}\n\n"
        f"{truncated}"
    )


def query_lmstudio(
    model_name: str | None,
    url: str,
    page_text: str,
    local_tracks: list[dict],
) -> AlbumMetadata:
    with lms.Client() as client:
        model = client.llm.model(
            model_name) if model_name else client.llm.model()
        chat = lms.Chat(SYSTEM_PROMPT)
        chat.add_user_message(build_user_message(url, page_text, local_tracks))
        result = model.respond(chat, response_format=AlbumMetadata)
    parsed = result.parsed
    meta = parsed if isinstance(
        parsed, AlbumMetadata) else AlbumMetadata.model_validate(parsed)
    return clean_metadata(meta)


def write_tags(audio_dir: Path, meta: AlbumMetadata) -> None:
    files = sorted(p for p in audio_dir.iterdir()
                   if p.suffix.lower() in AUDIO_EXTS)
    if len(files) != len(meta.tracks):
        print(
            f"WARNING: {len(files)} audio files vs {len(meta.tracks)} parsed tracks; "
            f"refusing to write. Inspect the JSON and re-run with corrected input.",
            file=sys.stderr,
        )
        return
    for path, track in zip(files, meta.tracks):
        f = mutagen.File(path, easy=True)
        if f is None:
            print(
                f"Skip (unsupported tag format): {path.name}", file=sys.stderr)
            continue
        f["title"] = track.title
        f["artist"] = track.artist or meta.album_artist
        f["album"] = meta.album
        f["albumartist"] = meta.album_artist
        f["tracknumber"] = str(track.track_number)
        f["discnumber"] = str(track.disc_number)
        if meta.date:
            f["date"] = meta.date
        if meta.genre:
            f["genre"] = meta.genre
        if track.composer:
            f["composer"] = track.composer
        f.save()
        print(f"Tagged: {path.name}")


def strip_disambiguator(title: str) -> str:
    m = TRAILING_PAREN_RE.search(title)
    if m and m.group(1).strip().lower() in WIKI_DISAMBIGUATORS:
        return title[: m.start()].strip()
    return title


def clean_metadata(meta: AlbumMetadata) -> AlbumMetadata:
    meta.album = strip_disambiguator(meta.album)
    for t in meta.tracks:
        t.title = strip_disambiguator(t.title)
    return meta


def sanitize(name: str) -> str:
    cleaned = INVALID_FS_CHARS.sub(" - ", name).strip().rstrip(". ").strip()
    return cleaned or "Untitled"


def rename_release(audio_dir: Path, meta: AlbumMetadata) -> Path | None:
    """Rename the directory and its files per "%A - %T (%y) [%f]/%n. %t"."""
    files = sorted(p for p in audio_dir.iterdir()
                   if p.suffix.lower() in AUDIO_EXTS)
    if len(files) != len(meta.tracks):
        print(
            f"WARNING: {len(files)} files vs {len(meta.tracks)} tracks; refusing to rename.",
            file=sys.stderr,
        )
        return None

    year = (meta.date or "")[:4] or "Unknown"
    fmt = files[0].suffix.lstrip(".").upper()
    dir_name = sanitize(f"{meta.album_artist} - {meta.album} ({year}) [{fmt}]")
    target_dir = audio_dir.parent / dir_name

    if target_dir.exists() and target_dir.resolve() != audio_dir.resolve():
        print(
            f"WARNING: target directory already exists: {target_dir}; refusing to overwrite.",
            file=sys.stderr,
        )
        return None

    if target_dir.resolve() != audio_dir.resolve():
        audio_dir.rename(target_dir)
        print(
            f"Renamed dir: {audio_dir.name} -> {target_dir.name}", file=sys.stderr)

    width = max(2, len(str(len(meta.tracks))))
    files_now = sorted(p for p in target_dir.iterdir()
                       if p.suffix.lower() in AUDIO_EXTS)
    for path, track in zip(files_now, meta.tracks):
        base = sanitize(f"{track.track_number:0{width}d}. {track.title}")
        new_path = target_dir / (base + path.suffix.lower())
        if new_path == path:
            continue
        if new_path.exists():
            print(
                f"WARNING: target exists, skipping: {new_path.name}", file=sys.stderr)
            continue
        path.rename(new_path)
        print(f"Renamed: {path.name} -> {new_path.name}", file=sys.stderr)
    return target_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True,
                    help="Webpage describing the release")
    ap.add_argument(
        "--dir", required=True, type=Path, help="Directory of ripped audio files"
    )
    ap.add_argument(
        "--model",
        default="mlx-qwen3.5-9b-glm5.1-distill-v1@8bit",
        help="LM Studio model identifier (default: currently loaded model)",
    )
    ap.add_argument("--out", type=Path, help="Write parsed JSON to this path")
    ap.add_argument(
        "--write", action="store_true", help="Write tags to the audio files"
    )
    ap.add_argument(
        "--rename",
        action="store_true",
        help='Rename dir and files to "%%A - %%T (%%y) [%%f]/%%n. %%t"',
    )
    args = ap.parse_args()

    if not args.dir.is_dir():
        sys.exit(f"Not a directory: {args.dir}")

    print(f"Scanning {args.dir} ...", file=sys.stderr)
    local_tracks = load_local_tracks(args.dir)
    print(f"Found {len(local_tracks)} audio file(s).", file=sys.stderr)

    print(f"Fetching {args.url} ...", file=sys.stderr)
    page_text = fetch_page_text(args.url)

    print("Querying LM Studio ...", file=sys.stderr)
    meta = query_lmstudio(args.model, args.url, page_text, local_tracks)

    out_json = meta.model_dump_json(indent=2)
    print(out_json)
    if args.out:
        args.out.write_text(out_json)
        print(f"Wrote {args.out}", file=sys.stderr)
    if args.write:
        write_tags(args.dir, meta)
    if args.rename:
        rename_release(args.dir, meta)


if __name__ == "__main__":
    main()
