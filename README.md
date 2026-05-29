# cd-metadata

Generate metadata for a ripped CD from a webpage URL using a locally-running LM Studio model.

Useful when MusicBrainz/Discogs don't have the release: point this at any page that describes it (label site, Bandcamp, artist page, Wikipedia, blog review) and the model extracts structured metadata aligned to your actual ripped tracks.

## Requirements

- [LM Studio](https://lmstudio.ai) running with a model loaded
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```sh
uv sync
```

## Usage

```sh
uv run cd_metadata.py --url URL --dir PATH_TO_RIPS
```

Writes the parsed metadata to `<dir>/metadata.json` and prints a
human-readable summary to stderr, including a `Status:` line that says
either `complete — ready to --write/--rename` or `needs review —
missing ...` so you can tell at a glance whether the extraction is
usable as-is or needs hand-editing.

### Flags

Exactly one source flag is required:

- `--url URL` — webpage describing the release
- `--image PATH [PATH ...]` — one or more cover/insert photos (vision-capable model required)
- `--metadata` — reuse `<dir>/metadata.json` from a prior run (no LM Studio call)

Plus:

- `--dir` — directory of ripped audio files (required)
- `--model` — LM Studio model identifier (defaults to a placeholder; pass `--model` to override or rely on whatever is currently loaded)
- `--out PATH` — write parsed JSON to this path instead of `<dir>/metadata.json`
- `--write` — write tags onto the audio files
- `--rename` — rename the directory to `%A - %T (%y) [%f]` and each file to `%n. %t`

`--write` and `--rename` both refuse to act if `album`, `album_artist`,
or any track title is null — the `Status:` line will say `needs
review` in that case. Edit `metadata.json` to fill in the missing
fields, then re-run with `--metadata` to apply.

### Examples

Preview metadata without changing files:

```sh
uv run cd_metadata.py --url https://example.com/release --dir ~/rips/disc1
```

Tag files and rename in place:

```sh
uv run cd_metadata.py --url https://example.com/release --dir ~/rips/disc1 --write --rename
```

Save the parsed JSON to a different path:

```sh
uv run cd_metadata.py --url https://example.com/release --dir ~/rips/disc1 --out tags.json
```

Re-apply edits from a hand-corrected `metadata.json` (no LM Studio call):

```sh
uv run cd_metadata.py --metadata --dir ~/rips/disc1 --write --rename
```

### Image mode

```sh
uv run cd_metadata.py --image PATH [PATH ...] --dir PATH_TO_RIPS
```

Pass one or more photos (back cover, insert, label) to a vision-capable
LM Studio model. Multiple images are treated as different views of the
same release — a field unreadable on the back cover may be clearly
legible on the insert. JPEG, PNG, WebP, and HEIC/HEIF are all accepted.

Every image is routed through macOS `sips` before upload: HEIC is
converted to JPEG, EXIF orientation is baked into pixels (so sideways
iPhone photos OCR correctly), and any dimension over 2048px is downscaled.

## Notes

- The URL flow runs in two LLM passes — album-level fields first, then the tracklist with the album context already locked in — to dodge the null-bias local quantized models exhibit under structured output.
- Discogs URLs (`/release/...`, `/master/...`) are fetched via the public JSON API since the site blocks scraping.
- Supported audio formats: FLAC, WAV, MP3, M4A, AIFF, OGG, Opus.
- Image mode requires macOS (`sips`).
- If the page's tracklist count doesn't match your rip, the model fills what it can and notes the discrepancy rather than guessing.
