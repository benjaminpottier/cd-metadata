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

    # Or, send one or more photos (back cover, liner notes, insert, label)
    # to a vision-capable LM Studio model:
    uv run cd_metadata.py --image PATH_TO_IMAGE [PATH_TO_IMAGE ...] --dir PATH_TO_RIPS

    # Second pass — apply tags and/or rename using the metadata from the first
    # pass (no LM Studio call this time):
    uv run cd_metadata.py --metadata --dir PATH_TO_RIPS --write --rename

Exactly one of --url (fetch & extract via LM Studio), --image (vision model
on one or more JPEG/PNG/WebP/HEIC files), or --metadata (reuse the existing
<dir>/metadata.json from a prior run) is required.

All --image inputs are routed through macOS `sips` before being sent to
LM Studio: HEIC/HEIF are converted to JPEG, oversized images are capped
at 2048px on the longest side, and EXIF orientation is baked into pixels
(so sideways iPhone shots OCR correctly). Non-macOS hosts must supply
already-oriented JPEG/PNG/WebP and will hit a clean exit if sips is missing.

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
    total = len(meta.tracks)
    titled = sum(1 for t in meta.tracks if t.title)
    if total == 0:
        tracks_line = "0"
    elif titled == total:
        tracks_line = f"{total} (all titled)"
    else:
        tracks_line = f"{total} ({titled} titled, {total - titled} missing)"

    # `write_tags` and `rename_release` both refuse to act when album,
    # album_artist, or any track title is null — so those are exactly the
    # fields whose presence the status check reports on.
    missing = []
    if not meta.album:
        missing.append("album")
    if not meta.album_artist:
        missing.append("album_artist")
    gap = total - titled
    if gap:
        missing.append(f"{gap} track title{'s' if gap != 1 else ''}")
    status = (
        "complete — ready to --write/--rename"
        if not missing
        else "needs review — missing " + ", ".join(missing)
    )

    summary_logger.info(
        "Album:        %s\n"
        "Album artist: %s\n"
        "Year:         %s\n"
        "Label:        %s\n"
        "Tracks:       %s\n"
        "Wrote:        %s\n"
        "Status:       %s",
        meta.album or "-",
        meta.album_artist or "-",
        year,
        meta.label or "-",
        tracks_line,
        out_path,
        status,
    )
    if meta.notes:
        indented = "\n".join(f"  {line}" for line in meta.notes.splitlines())
        summary_logger.info("Notes:\n%s", indented)

AUDIO_EXTS = {".flac", ".wav", ".mp3",
              ".m4a", ".aiff", ".aif", ".ogg", ".opus"}
# Image extensions LM Studio vision models reliably ingest directly.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# HEIC/HEIF are accepted at the CLI but converted to JPEG via macOS `sips`
# (vision models don't reliably handle HEIC).
HEIC_EXTS = {".heic", ".heif"}
# All accepted images are routed through sips for normalization (EXIF
# orientation bake-in + longest-side cap). Vision-model OCR fails silently
# on sideways photos that carry only an EXIF rotation flag, and uploading
# an unshrunk 12MP iPhone photo wastes both bandwidth and inference time.
IMAGE_MAX_DIMENSION = 2048
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
# Fields kept from the Discogs API response. Everything else (community
# ratings, submitter info, image URLs, resource_url links, etc.) is noise
# that dilutes the model's attention without helping extraction.
DISCOGS_KEEP_KEYS = {
    "title", "artists", "artists_sort", "year", "released", "released_formatted",
    "labels", "companies", "formats", "genres", "styles", "country",
    "tracklist", "identifiers", "extraartists", "notes",
}
USER_AGENT = "cd-metadata/0.1 (+https://github.com/local/cd-metadata)"


class Track(BaseModel):
    track_number: int
    disc_number: int = 1
    # Optional so the model can return null per-track when illegible (image
    # mode) or absent (text mode), instead of confabulating a title. Editable
    # by the user via the metadata.json + --metadata workflow.
    title: str | None = None
    artist: str | None = None
    composer: str | None = None
    duration_seconds: float | None = None


class AlbumMetadata(BaseModel):
    # `album` and `album_artist` are Optional so the model is permitted to
    # admit it cannot read these from the source. With them required, the
    # response_format=AlbumMetadata schema forced the model to invent values
    # even when the image/page didn't contain them. Downstream code
    # (write_tags, rename_release) refuses to act on null required fields.
    album: str | None = None
    album_artist: str | None = None
    date: str | None = None
    label: str | None = None
    catalog_number: str | None = None
    genre: str | None = None
    tracks: list[Track]
    notes: str | None = None


# Two-stage extraction for the URL flow: a small schema for album-level
# scalars, then a small schema for tracks + alignment. Splitting the model's
# job in two reduces the per-call cognitive load — under structured output,
# local quantized models tend to default to null on optional string fields
# when the schema is large (8 fields + a tracks array). With just the
# scalars to fill, the model populates `album`/`album_artist` reliably.
class AlbumScalars(BaseModel):
    album: str | None = None
    album_artist: str | None = None
    date: str | None = None
    label: str | None = None
    catalog_number: str | None = None
    genre: str | None = None
    notes: str | None = None


class TrackTitlesAndNotes(BaseModel):
    """LLM contract for the tracks pass.

    `track_titles` is `list[str]` (not `list[Track]` and not `list[str | None]`)
    on purpose. With `list[Track]`, the per-track `title: str | None` field
    triggered constrained-decoding null bias on local quantized models —
    every track in a release came back with `title: null` even when the
    page clearly stated all titles. A required string with an explicit ""
    sentinel for "not stated" gives the model a clear escape hatch without
    the JSON schema biasing every position toward null. Python maps "" back
    to None before building the final `Track` objects.
    """
    track_titles: list[str]
    notes: str | None = None


SCALAR_SYSTEM_PROMPT = """You extract album-level metadata from a webpage.

Fill these fields:
- album: the album title
- album_artist: the album's primary artist
- date: release date as "YYYY" or "YYYY-MM-DD" (only if the page is that specific)
- label: the record label
- catalog_number: the label's catalog number
- genre: musical genre

Rules:
- The page is your source of truth. Look everywhere — main prose, sidebars, infoboxes, tables, captions, JSON-LD blocks, structured API responses. If a value appears anywhere on the page, transcribe it.
- The page may be a structured API response (e.g., Discogs JSON). Field names there may not match the schema exactly: an album's title may be a top-level `title`, the artist may sit inside `artists[].name`, the release date may be `released` or `year`. Map those to the schema fields above.
- Return null ONLY when a field is genuinely absent from the page. "Buried in an infobox" or "phrased awkwardly" is NOT a reason to return null.
- If you return null for `album` or `album_artist`, describe in `notes` what the page DID say (e.g., "page is an artist bio that mentions the label and year but never names this specific album").
- Never invent values.
- Strip Wikipedia/CMS disambiguator suffixes from `album`. These are single short descriptors in trailing parentheses — "(album)", "(EP)", "(song)", "(soundtrack)", "(text)", "(disambiguation)", "(film)", "(band)", etc. They are page-routing artifacts, not part of the real title. Preserve legitimate parentheticals that are part of the title itself (e.g., "(What's the Story) Morning Glory?", "Smells Like Teen Spirit (Single Edit)").
"""

TRACKS_SYSTEM_PROMPT = """You extract a tracklist from a webpage and align it to a user's ripped CD tracks.

Output a JSON object:
- track_titles: an array of strings, ONE entry per user-ripped track, in order.
- notes: optional commentary on alignment, missing titles, or duration mismatches.

Rules:
- Read the tracklist from the page. Look at lists, tables, JSON tracklist arrays, anywhere track titles are stated.
- For each of the user's ripped track positions, output the corresponding track title from the page at the same index in `track_titles`.
- The length of `track_titles` MUST equal the number of ripped tracks the user provides. Do not skip entries; do not concatenate or merge tracks.
- The user's filenames are for ALIGNMENT only. Placeholder names like "Unknown Artist" or "01.flac" are NOT evidence that titles are unknown — read the page.
- If a particular position has no title on the page (e.g., the page lists fewer tracks than the rip, or a position is illegible/absent), emit an EMPTY STRING "" for that entry. "" means "not stated on the page". Never invent a value.
- If the page's tracklist count does not match the rip's count, fill what you can and explain the discrepancy in `notes`.
- If the page lists durations, verify alignment; flag mismatches greater than 5 seconds in `notes`.
- Strip Wikipedia/CMS disambiguator suffixes from track titles. These are single short descriptors in trailing parentheses — "(song)", "(track)", "(disambiguation)", etc. Preserve legitimate parentheticals that are part of the real title (e.g., "Smells Like Teen Spirit (Single Edit)").
"""

IMAGE_SYSTEM_PROMPT = """You extract album metadata from one or more photos of a CD release (back cover, liner notes, insert, label, etc.) and align it to a user's ripped CD tracks.

Core rules:
- The image(s) are your source of truth. Transcribe what you can read from them. If text required some effort to read but you can read it, transcribe it — do not return null just because OCR was not effortless.
- When multiple images are provided, treat them as different views of the SAME release (e.g., back cover + insert + booklet page). Combine information across them; a field unreadable in one image may be clearly legible in another.
- Return null ONLY when a field is not present in any image (e.g., none of the photos list a year) or when the text is genuinely unreadable across all of them (fully covered, cropped off, severely damaged). "Not perfectly clear" is NOT a reason to return null.
- Do not invent values that are not present in the image(s). If a field is genuinely absent or unreadable, return null and briefly explain in `notes`.
- The user's ripped track list (filenames and durations) is for ALIGNMENT only. Use it to match positions to the tracklist you read FROM THE IMAGE(S) — do not use track count or durations to identify the album or fill in track titles you cannot actually read.

Workflow:
1. Read every image. Identify the visible text across all of them: album title, artist, track titles, label, year, catalog number.
2. Transcribe each visible field. Use null only for fields not present in any image or text you genuinely cannot read.
3. Align the visible tracklist to the user's ripped track positions, IN ORDER.

Additional rules:
- If the visible tracklist count does not match the rip's track count, fill what you can and explain the discrepancy in `notes`.
- If durations are listed, verify alignment; flag mismatches greater than 5 seconds in `notes`.
- If a portion of the tracklist is genuinely unreadable (covered, cropped, damaged across all images), return null for those track titles and note the gap in `notes`.
- If you return null for `album` or `album_artist`, describe in `notes` what you DID see (e.g., "back cover with white text on black, label visible as 'XYZ Records', album title torn off").
- Emit track_number starting at 1, matching the order of the user's files.
- Dates: "YYYY" or "YYYY-MM-DD" only if an image is that specific.
"""


def _audio_files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO_EXTS)


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

    soup = BeautifulSoup(raw, "html.parser")
    ld_blocks: list[str] = []
    for s in soup.find_all("script", type="application/ld+json"):
        if s.string:
            ld_blocks.append(s.string.strip())

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    body_text = soup.get_text("\n", strip=True)

    parts = []
    if ld_blocks:
        parts.append("=== JSON-LD ===\n" + "\n---\n".join(ld_blocks))
    parts.append("=== PAGE TEXT ===\n" + body_text)
    return "\n\n".join(parts)


def _truncate_page(page_text: str) -> str:
    truncated = page_text[:PAGE_CHAR_BUDGET]
    if len(page_text) > PAGE_CHAR_BUDGET:
        truncated += f"\n[... {len(page_text) - PAGE_CHAR_BUDGET} more chars truncated]"
    return truncated


def build_scalar_message(url: str, page_text: str) -> str:
    """User message for the album-scalars pass — page text, no filenames.

    The scalars pass doesn't need the user's track list; including it would
    only add noise. Filenames like "Unknown Artist" can prime the model to
    return null even when the page clearly states the album.
    """
    return (
        f"Source URL: {url}\n\n"
        f"Below is the page describing an album release. "
        f"Extract the album-level fields.\n\n"
        f"{_truncate_page(page_text)}"
    )


def build_tracks_message(
    url: str,
    page_text: str,
    local_tracks: list[dict],
    scalars: AlbumScalars,
) -> str:
    """User message for the tracks pass — page first, then alignment list.

    The album-level scalars already extracted by pass 1 are passed back as
    confirmed context so the model knows which release it's reading and
    isn't tempted to second-guess the album identity from the page.
    """
    track_list = "\n".join(
        f"  {t['position']}. {t['filename']} ({t['duration_seconds']}s)"
        for t in local_tracks
    )
    context_parts = []
    if scalars.album:
        context_parts.append(f"  album: {scalars.album}")
    if scalars.album_artist:
        context_parts.append(f"  album_artist: {scalars.album_artist}")
    if scalars.date:
        context_parts.append(f"  date: {scalars.date}")
    context_block = (
        "Album context (already confirmed in a prior step, use to "
        "disambiguate the tracklist):\n" + "\n".join(context_parts) + "\n\n"
        if context_parts else ""
    )
    return (
        f"Source URL: {url}\n\n"
        f"{context_block}"
        f"Below is the page describing the release. Extract the tracklist.\n\n"
        f"{_truncate_page(page_text)}\n\n"
        f"User's {len(local_tracks)} ripped track(s) — for ALIGNMENT only. "
        f"Match the tracklist you read above to these positions in order. "
        f"The filenames may be placeholders (e.g., 'Unknown'); do not let "
        f"them override the values you read from the page:\n{track_list}"
    )


def query_lmstudio(
    model_name: str | None,
    url: str,
    page_text: str,
    local_tracks: list[dict],
) -> AlbumMetadata:
    """Two-stage extraction: scalars first, then tracks + alignment notes.

    Local quantized models running under structured output tend to default
    to null on optional string fields when the response schema is large.
    Splitting into two smaller schemas (album scalars; tracks+notes) keeps
    each call's cognitive load light and dramatically improves recall of
    `album` and `album_artist`. Notes from both stages are concatenated.
    """
    with lms.Client() as client:
        model = client.llm.model(
            model_name) if model_name else client.llm.model()

        chat = lms.Chat(SCALAR_SYSTEM_PROMPT)
        chat.add_user_message(build_scalar_message(url, page_text))
        scalars_result = model.respond(chat, response_format=AlbumScalars)
        scalars_parsed = scalars_result.parsed
        scalars = scalars_parsed if isinstance(
            scalars_parsed, AlbumScalars
        ) else AlbumScalars.model_validate(scalars_parsed)

        chat = lms.Chat(TRACKS_SYSTEM_PROMPT)
        chat.add_user_message(
            build_tracks_message(url, page_text, local_tracks, scalars)
        )
        tracks_result = model.respond(chat, response_format=TrackTitlesAndNotes)
        tracks_parsed = tracks_result.parsed
        tan = tracks_parsed if isinstance(
            tracks_parsed, TrackTitlesAndNotes
        ) else TrackTitlesAndNotes.model_validate(tracks_parsed)

    # Build the final Track list. The model returns "" for "not stated" —
    # we map that back to None so write_tags/rename_release's guard fires
    # the same way it would for any other null title. If the model emitted
    # fewer titles than ripped positions, pad with title=None Tracks; if
    # more, truncate to the rip's count (the prompt forbids this but local
    # models occasionally miscount and the alignment must stay 1:1 with
    # the audio files for downstream tagging).
    expected = len(local_tracks) if local_tracks else len(tan.track_titles)
    raw_titles = list(tan.track_titles[:expected])
    raw_titles += [""] * (expected - len(raw_titles))
    tracks = [
        Track(track_number=i + 1, title=(t.strip() or None))
        for i, t in enumerate(raw_titles)
    ]

    notes_parts = [n for n in (scalars.notes, tan.notes) if n]
    merged_notes = "\n".join(notes_parts) or None
    meta = AlbumMetadata(
        album=scalars.album,
        album_artist=scalars.album_artist,
        date=scalars.date,
        label=scalars.label,
        catalog_number=scalars.catalog_number,
        genre=scalars.genre,
        tracks=tracks,
        notes=merged_notes,
    )
    return clean_metadata(meta)


def build_image_chat_text(
    local_tracks: list[dict], num_images: int = 1
) -> tuple[str, str]:
    """Returns (intro_before_images, alignment_after_images).

    The image(s) are attached BETWEEN these two text parts so the model
    reads instructions, then sees the image(s), then receives the alignment
    data. Putting the track list AFTER the images discourages the model
    from using durations as an album fingerprint before looking at them.
    """
    track_list = "\n".join(
        f"  {t['position']}. {t['filename']} ({t['duration_seconds']}s)"
        for t in local_tracks
    )
    if num_images == 1:
        image_phrase = "Below is an image of an album release."
        align_phrase = "match the tracklist you read from the image to those"
        invent_phrase = "track titles that are not visible in the image."
        align_intro = "the tracklist you read from the image"
    else:
        image_phrase = (
            f"Below are {num_images} images of an album release "
            "(e.g., different panels of the back cover, insert, or label)."
        )
        align_phrase = (
            "match the tracklist you read across the images to those"
        )
        invent_phrase = "track titles that are not visible in any image."
        align_intro = "the tracklist you read from the images"
    intro = (
        f"{image_phrase} Transcribe the metadata you can read.\n\n"
        f"The user has {len(local_tracks)} ripped track(s); their filenames "
        "and durations follow AFTER the image(s). That list is for ALIGNMENT "
        f"only — {align_phrase} positions. Do not use the durations, "
        f"filenames, or count to invent an album identity or {invent_phrase}"
    )
    alignment = (
        f"User's {len(local_tracks)} ripped track(s) "
        f"(align these positions to {align_intro}):\n"
        f"{track_list}"
    )
    return intro, alignment


def _run_sips(args: list[str]) -> subprocess.CompletedProcess:
    """Invoke sips with a consistent FileNotFoundError exit.

    All sips invocations route through here so the macOS-required message
    is the same wherever sips is needed.
    """
    try:
        return subprocess.run(
            ["sips", *args], capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        sys.exit(
            "sips not found — image normalization requires macOS. "
            "On other platforms, convert/orient externally and rerun."
        )


def _image_needs_resize(image_path: Path) -> bool:
    """True iff sips reports either pixel dimension > IMAGE_MAX_DIMENSION.

    On any failure to read dimensions (corrupt header, unexpected output),
    return True so the subsequent conversion call surfaces the real error
    via its own returncode check.
    """
    result = _run_sips(["-g", "pixelWidth", "-g", "pixelHeight", str(image_path)])
    if result.returncode != 0:
        return True
    for line in result.stdout.splitlines():
        key, _, value = line.strip().partition(":")
        if key in ("pixelWidth", "pixelHeight"):
            try:
                if int(value.strip()) > IMAGE_MAX_DIMENSION:
                    return True
            except ValueError:
                return True
    return False


@contextlib.contextmanager
def _resolve_image(image_path: Path) -> Iterator[Path]:
    """Yield a normalized JPEG temp path the LM Studio SDK can ingest.

    Every accepted image is routed through macOS `sips` to convert to JPEG
    and bake EXIF orientation into pixels. Images whose longest pixel
    dimension exceeds IMAGE_MAX_DIMENSION are also downsized (sips' -Z
    flag is bidirectional, so we gate it with a dimension check rather
    than letting it upscale small back-cover photos). The temp file is
    cleaned up on exit, including on exceptions.
    """
    fd, tmp_name = tempfile.mkstemp(suffix=".jpg", prefix="cd-metadata-")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        cmd = ["-s", "format", "jpeg"]
        if _image_needs_resize(image_path):
            cmd += ["-Z", str(IMAGE_MAX_DIMENSION)]
        cmd += [str(image_path), "--out", str(tmp_path)]
        result = _run_sips(cmd)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip() or "(no message)"
            sys.exit(f"sips failed to convert {image_path}: {detail}")
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def query_lmstudio_image(
    model_name: str | None,
    image_paths: list[Path],
    local_tracks: list[dict],
) -> AlbumMetadata:
    intro, alignment = build_image_chat_text(local_tracks, num_images=len(image_paths))
    with lms.Client() as client:
        handles = [client.prepare_image(p) for p in image_paths]
        model = client.llm.model(
            model_name) if model_name else client.llm.model()
        chat = lms.Chat(IMAGE_SYSTEM_PROMPT)
        # Image(s) attached between the two text parts. The lmstudio SDK
        # accepts a list mixing text and FileHandle as `content`; this is
        # the only way to position images in the middle of the user turn.
        chat.add_user_message([intro, *handles, alignment])
        result = model.respond(chat, response_format=AlbumMetadata)
    parsed = result.parsed
    meta = parsed if isinstance(
        parsed, AlbumMetadata) else AlbumMetadata.model_validate(parsed)
    return clean_metadata(meta)


def _metadata_is_incomplete(meta: AlbumMetadata) -> bool:
    return (
        meta.album is None
        or meta.album_artist is None
        or any(t.title is None for t in meta.tracks)
    )


def write_tags(audio_dir: Path, meta: AlbumMetadata) -> None:
    files = _audio_files(audio_dir)
    if len(files) != len(meta.tracks):
        logger.warning(
            "%d audio files vs %d parsed tracks; refusing to write. "
            "Inspect the JSON and re-run with corrected input.",
            len(files), len(meta.tracks),
        )
        return
    if _metadata_is_incomplete(meta):
        logger.warning(
            "album, album_artist, or a track title is null; refusing to "
            "write tags. Edit metadata.json to fill them in and retry."
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
    if meta.album is not None:
        meta.album = strip_disambiguator(meta.album)
    for t in meta.tracks:
        if t.title is not None:
            t.title = strip_disambiguator(t.title)
    return meta


def sanitize(name: str) -> str:
    cleaned = INVALID_FS_CHARS.sub(" - ", name).strip().rstrip(". ").strip()
    return cleaned or "Untitled"


def rename_release(audio_dir: Path, meta: AlbumMetadata) -> Path | None:
    """Rename the directory and its files per "%A - %T (%y) [%f]/%n. %t"."""
    files = _audio_files(audio_dir)
    if len(files) != len(meta.tracks):
        logger.warning(
            "%d files vs %d tracks; refusing to rename.",
            len(files), len(meta.tracks),
        )
        return None
    if _metadata_is_incomplete(meta):
        logger.warning(
            "album, album_artist, or a track title is null; refusing to "
            "rename. Edit metadata.json to fill them in and retry."
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
    files_now = _audio_files(target_dir)
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
        nargs="+",
        metavar="PATH",
        help="One or more image files (back cover, liner notes, insert); "
        "requires a vision-capable model. JPEG/PNG/WebP/HEIC. Multiple "
        "images are treated as different views of the same release.",
    )
    ap.add_argument(
        "--dir", required=True, type=Path, help="Directory of ripped audio files"
    )
    ap.add_argument(
        "--model",
        default="qwen2.5-vl-32b-instruct",
        help="LM Studio model identifier (default: qwen2.5-vl-32b-instruct, "
        "validated to handle both URL and image modes well)",
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
        for img in args.image:
            if not img.is_file():
                sys.exit(f"Not a file: {img}")
            if img.suffix.lower() not in (IMAGE_EXTS | HEIC_EXTS):
                sys.exit(
                    f"Unsupported image format: {img.suffix} "
                    "(JPEG, PNG, WebP, HEIC/HEIF only)"
                )
        logger.info("Scanning %s ...", args.dir)
        local_tracks = load_local_tracks(args.dir)
        logger.info("Found %d audio file(s).", len(local_tracks))

        with contextlib.ExitStack() as stack:
            resolved = [stack.enter_context(_resolve_image(p)) for p in args.image]
            for src, dst in zip(args.image, resolved):
                logger.info("Normalized %s -> %s (sips)", src, dst)
            logger.info(
                "Sending %d image(s) to LM Studio ...", len(resolved),
            )
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
