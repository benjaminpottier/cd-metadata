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
    # First pass — generate metadata for review:
    uv run cd_metadata.py --url URL --dir PATH_TO_RIPS

    # Or, send a photo of the back cover / liner notes to a vision-capable
    # LM Studio model:
    uv run cd_metadata.py --image PATH_TO_IMAGE --dir PATH_TO_RIPS

    # Second pass — apply tags and/or rename using the metadata from the first
    # pass (no LM Studio call this time):
    uv run cd_metadata.py --metadata --dir PATH_TO_RIPS --write --rename

Exactly one of --url (fetch & extract via LM Studio), --image (vision model
on a JPEG/PNG/WebP/HEIC), or --metadata (reuse the existing
<dir>/metadata.json from a prior run) is required.

HEIC/HEIF inputs are transparently converted to JPEG via macOS `sips`
before being sent to LM Studio (so iPhone photos work out of the box on a
Mac). Non-macOS hosts should convert externally.

Output:
    The parsed metadata is always written to a JSON file. Without --out, it
    lands at "<dir>/metadata.json" (following --rename to the new dir name if
    used). A human-readable summary of the run is printed to stderr.

With --rename, the directory is renamed to "%A - %T (%y) [%f]" and each
file to "%n. %t" preserving its original extension.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import lmstudio as lms
import mutagen
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

logger = logging.getLogger(__name__)
# Separate logger for the human-readable end-of-run summary. Writes raw lines
# (no "INFO:" prefix) to stderr via its own handler, and does not propagate to
# the root logger so the basicConfig handler does not also prefix-emit them.
summary_logger = logging.getLogger(f"{__name__}.summary")


class _LateStderrHandler(logging.Handler):
    """StreamHandler that resolves sys.stderr at emit time, not construction.

    Without this, tests that swap sys.stderr (pytest's capsys) would miss the
    summary because StreamHandler captures the stream reference at __init__.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
        except Exception:  # pragma: no cover - logging fallback
            self.handleError(record)


class _ColorFormatter(logging.Formatter):
    """Wraps the level prefix in ANSI color when enabled. Message text is left
    untouched so file paths and identifiers keep their default color.
    """

    _COLORS = {
        logging.DEBUG: "\033[2m",       # dim
        logging.INFO: "\033[36m",       # cyan
        logging.WARNING: "\033[33m",    # yellow
        logging.ERROR: "\033[31m",      # red
        logging.CRITICAL: "\033[31;1m",  # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, fmt: str, *, color: bool) -> None:
        super().__init__(fmt)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        code = self._COLORS.get(record.levelno) if self.color else None
        if code is None:
            return super().format(record)
        original = record.levelname
        record.levelname = f"{code}{original}{self._RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def _color_enabled() -> bool:
    """Color on iff stderr is a TTY and NO_COLOR is unset (https://no-color.org)."""
    return sys.stderr.isatty() and not os.environ.get("NO_COLOR")


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            _ColorFormatter("%(levelname)s: %(message)s", color=_color_enabled())
        )
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    if not summary_logger.handlers:
        handler = _LateStderrHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        summary_logger.addHandler(handler)
        summary_logger.setLevel(logging.INFO)
        summary_logger.propagate = False


def log_summary(meta: AlbumMetadata, out_path: Path) -> None:
    year = (meta.date or "")[:4] or "-"
    summary_logger.info(
        "Album:        %s\n"
        "Album artist: %s\n"
        "Year:         %s\n"
        "Label:        %s\n"
        "Tracks:       %d\n"
        "Wrote:        %s",
        meta.album,
        meta.album_artist,
        year,
        meta.label or "-",
        len(meta.tracks),
        out_path,
    )
    if meta.notes:
        indented = "\n".join(f"  {line}" for line in meta.notes.splitlines())
        summary_logger.info("Notes:\n%s", indented)

AUDIO_EXTS = {".flac", ".wav", ".mp3",
              ".m4a", ".aiff", ".aif", ".ogg", ".opus"}
# Image extensions LM Studio vision models reliably ingest directly.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# HEIC/HEIF are accepted at the CLI but converted to JPEG via macOS `sips`
# before being sent to LM Studio (vision models don't reliably handle HEIC).
HEIC_EXTS = {".heic", ".heif"}
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

IMAGE_SYSTEM_PROMPT = """You extract album metadata from a photo of a CD release (back cover, liner notes, insert, label, etc.) and align it to a user's ripped CD tracks.

Rules:
- Use ONLY information visible in the image. Never invent values.
- For any field not visible, return null.
- The user provides their actual track count and per-track durations in file order. Align the image's tracklist to those tracks IN ORDER.
- If the image's tracklist count does not match the rip's track count, fill what you can and explain the discrepancy in `notes`.
- If the image lists durations, verify alignment; flag mismatches greater than 5 seconds in `notes`.
- If part of the tracklist (or any other field) is illegible, obscured, cropped, or cut off, return null for that field and note the gap in `notes`.
- Emit track_number starting at 1, matching the order of the user's files.
- Dates: "YYYY" or "YYYY-MM-DD" only if the image is that specific.
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


def build_image_user_message(local_tracks: list[dict]) -> str:
    track_list = "\n".join(
        f"  {t['position']}. {t['filename']} ({t['duration_seconds']}s)"
        for t in local_tracks
    )
    return (
        f"My ripped CD has {len(local_tracks)} track(s) in this order:\n{track_list}\n\n"
        "Extract the album metadata from the attached image."
    )


@contextlib.contextmanager
def _resolve_image(image_path: Path) -> Iterator[Path]:
    """Yield a path the LM Studio SDK can ingest.

    HEIC/HEIF inputs are transparently converted to JPEG via macOS `sips`
    into a temp file that is cleaned up on exit. Other formats pass through.
    """
    if image_path.suffix.lower() not in HEIC_EXTS:
        yield image_path
        return

    fd, tmp_name = tempfile.mkstemp(suffix=".jpg", prefix="cd-metadata-")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        try:
            result = subprocess.run(
                ["sips", "-s", "format", "jpeg",
                 str(image_path), "--out", str(tmp_path)],
                capture_output=True, text=True, check=False,
            )
        except FileNotFoundError:
            sys.exit(
                "sips not found — HEIC/HEIF conversion requires macOS. "
                "Convert externally or pass JPEG/PNG/WebP."
            )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip() or "(no message)"
            sys.exit(f"sips failed to convert {image_path}: {detail}")
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def query_lmstudio_image(
    model_name: str | None,
    image_path: Path,
    local_tracks: list[dict],
) -> AlbumMetadata:
    with lms.Client() as client:
        handle = client.prepare_image(image_path)
        model = client.llm.model(
            model_name) if model_name else client.llm.model()
        chat = lms.Chat(IMAGE_SYSTEM_PROMPT)
        chat.add_user_message(
            build_image_user_message(local_tracks), images=[handle]
        )
        result = model.respond(chat, response_format=AlbumMetadata)
    parsed = result.parsed
    meta = parsed if isinstance(
        parsed, AlbumMetadata) else AlbumMetadata.model_validate(parsed)
    return clean_metadata(meta)


def write_tags(audio_dir: Path, meta: AlbumMetadata) -> None:
    files = sorted(p for p in audio_dir.iterdir()
                   if p.suffix.lower() in AUDIO_EXTS)
    if len(files) != len(meta.tracks):
        logger.warning(
            "%d audio files vs %d parsed tracks; refusing to write. "
            "Inspect the JSON and re-run with corrected input.",
            len(files), len(meta.tracks),
        )
        return
    for path, track in zip(files, meta.tracks):
        f = mutagen.File(path, easy=True)
        if f is None:
            logger.warning("Skip (unsupported tag format): %s", path.name)
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
        logger.info("Tagged: %s", path.name)


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
        logger.warning(
            "%d files vs %d tracks; refusing to rename.",
            len(files), len(meta.tracks),
        )
        return None

    year = (meta.date or "")[:4] or "Unknown"
    fmt = files[0].suffix.lstrip(".").upper()
    dir_name = sanitize(f"{meta.album_artist} - {meta.album} ({year}) [{fmt}]")
    target_dir = audio_dir.parent / dir_name

    if target_dir.exists() and target_dir.resolve() != audio_dir.resolve():
        logger.warning(
            "target directory already exists: %s; refusing to overwrite.",
            target_dir,
        )
        return None

    if target_dir.resolve() != audio_dir.resolve():
        audio_dir.rename(target_dir)
        logger.info("Renamed dir: %s -> %s", audio_dir.name, target_dir.name)

    width = max(2, len(str(len(meta.tracks))))
    files_now = sorted(p for p in target_dir.iterdir()
                       if p.suffix.lower() in AUDIO_EXTS)
    for path, track in zip(files_now, meta.tracks):
        base = sanitize(f"{track.track_number:0{width}d}. {track.title}")
        new_path = target_dir / (base + path.suffix.lower())
        if new_path == path:
            continue
        if new_path.exists():
            logger.warning("target exists, skipping: %s", new_path.name)
            continue
        path.rename(new_path)
        logger.info("Renamed: %s -> %s", path.name, new_path.name)
    return target_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Webpage describing the release")
    source.add_argument(
        "--metadata",
        action="store_true",
        help="Reuse <dir>/metadata.json from a prior run instead of calling LM Studio",
    )
    source.add_argument(
        "--image",
        type=Path,
        help="Image file (back cover, liner notes); requires a vision-capable model. "
        "JPEG/PNG/WebP only.",
    )
    ap.add_argument(
        "--dir", required=True, type=Path, help="Directory of ripped audio files"
    )
    ap.add_argument(
        "--model",
        default="mlx-qwen3.5-9b-glm5.1-distill-v1@8bit",
        help="LM Studio model identifier (default: currently loaded model)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        help="Write parsed JSON to this path (default: <dir>/metadata.json, "
        "following --rename if used)",
    )
    ap.add_argument(
        "--write", action="store_true", help="Write tags to the audio files"
    )
    ap.add_argument(
        "--rename",
        action="store_true",
        help='Rename dir and files to "%%A - %%T (%%y) [%%f]/%%n. %%t"',
    )
    args = ap.parse_args()

    _configure_logging()

    if not args.dir.is_dir():
        sys.exit(f"Not a directory: {args.dir}")

    if args.metadata:
        meta_path = args.dir / "metadata.json"
        if not meta_path.is_file():
            sys.exit(
                f"No metadata file at {meta_path} — run without --metadata first"
            )
        logger.info("Loading metadata from %s ...", meta_path)
        meta = clean_metadata(
            AlbumMetadata.model_validate_json(meta_path.read_text())
        )
    elif args.image:
        if not args.image.is_file():
            sys.exit(f"Not a file: {args.image}")
        if args.image.suffix.lower() not in (IMAGE_EXTS | HEIC_EXTS):
            sys.exit(
                f"Unsupported image format: {args.image.suffix} "
                "(JPEG, PNG, WebP, HEIC/HEIF only)"
            )
        logger.info("Scanning %s ...", args.dir)
        local_tracks = load_local_tracks(args.dir)
        logger.info("Found %d audio file(s).", len(local_tracks))

        with _resolve_image(args.image) as resolved:
            if resolved is not args.image:
                logger.info("Converted %s -> %s (sips)", args.image, resolved)
            logger.info("Sending image %s to LM Studio ...", resolved)
            meta = query_lmstudio_image(args.model, resolved, local_tracks)
    else:
        logger.info("Scanning %s ...", args.dir)
        local_tracks = load_local_tracks(args.dir)
        logger.info("Found %d audio file(s).", len(local_tracks))

        logger.info("Fetching %s ...", args.url)
        page_text = fetch_page_text(args.url)

        logger.info("Querying LM Studio ...")
        meta = query_lmstudio(args.model, args.url, page_text, local_tracks)

    if args.write:
        write_tags(args.dir, meta)

    audio_dir = args.dir
    if args.rename:
        renamed = rename_release(args.dir, meta)
        if renamed is not None:
            audio_dir = renamed

    out_path = args.out or (audio_dir / "metadata.json")
    out_path.write_text(meta.model_dump_json(indent=2))

    log_summary(meta, out_path)


if __name__ == "__main__":
    main()
