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

Prints the parsed metadata as JSON.

### Flags

- `--url` — webpage describing the release (required)
- `--dir` — directory of ripped audio files (required)
- `--model` — LM Studio model identifier (default: currently loaded model)
- `--out PATH` — write parsed JSON to this path
- `--write` — write tags onto the audio files
- `--rename` — rename the directory to `%A - %T (%y) [%f]` and each file to `%n. %t`

### Examples

Preview metadata without changing files:

```sh
uv run cd_metadata.py --url https://example.com/release --dir ~/rips/disc1
```

Tag files and rename in place:

```sh
uv run cd_metadata.py --url https://example.com/release --dir ~/rips/disc1 --write --rename
```

Save the parsed JSON for later:

```sh
uv run cd_metadata.py --url https://example.com/release --dir ~/rips/disc1 --out tags.json
```

## Notes

- Discogs URLs (`/release/...`, `/master/...`) are fetched via the public JSON API since the site blocks scraping.
- Supported audio formats: FLAC, WAV, MP3, M4A, AIFF, OGG, Opus.
- If the page's tracklist count doesn't match your rip, the model fills what it can and notes the discrepancy rather than guessing.
