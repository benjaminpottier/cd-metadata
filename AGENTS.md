# AGENTS.md

Guidance for AI coding agents working in this repo.

## What this is

A single-file Python CLI (`cd_metadata.py`) that generates album/track metadata
for a ripped CD by sending a webpage describing the release to a **locally
running LM Studio model** and aligning its structured output to the user's
actual audio files. Optional: write the tags into the files and rename the
directory/files to a canonical pattern.

The use case is for albums that MusicBrainz/Discogs doesn't have. The user
points at any web page (label site, Bandcamp, artist page, Wikipedia, blog
review) and the model does the extraction.

## Repo layout

- `cd_metadata.py` — the whole tool. One file, no package.
- `test_cd_metadata.py` — pytest suite at the repo root, paired with the tool.
- `pyproject.toml` — dependencies, managed by `uv`. Test deps live in the `dev` dependency group.
- `uv.lock` — pinned lockfile; commit changes when deps move.
- **Use `uv` for everything**: dependency management, running, testing. Don't forget it's required.
- `command.example` — example invocation.
- `.venv/` — created by `uv sync`. Do not commit.

There is intentionally no `src/` layout and no CI. Tests sit next to the tool and run with `pytest`.

## Running it

```
uv sync                        # one-time, installs into .venv

# First pass — text mode: LM Studio call, writes <dir>/metadata.json
uv run cd_metadata.py --url URL --dir PATH_TO_RIPS

# Or — image mode: send a back-cover photo to a vision-capable model
uv run cd_metadata.py --image PATH_TO_IMAGE --dir PATH_TO_RIPS

# Second pass: reuse the metadata, apply tags/rename without re-calling LM Studio
uv run cd_metadata.py --metadata --dir PATH_TO_RIPS --write --rename
```

Exactly one of `--url`, `--image`, or `--metadata` is required (argparse
mutually exclusive group). `--metadata` reads `<dir>/metadata.json` and
skips the network fetch + LM Studio call entirely. `--image` accepts
JPEG/PNG/WebP (`IMAGE_EXTS`) directly, plus HEIC/HEIF (`HEIC_EXTS`) which
are auto-converted via macOS `sips`.

## Running tests

```
uv sync --group dev                                           # one-time
uv run --group dev pytest                                     # run the suite
uv run --group dev pytest --cov=cd_metadata --cov-report=term-missing   # with coverage
```

Tests mock `requests`, `lmstudio`, and `mutagen.File` so they run hermetically — no
network and no LM Studio required. The suite currently holds 100% line + branch
coverage; keep it that way when changing `cd_metadata.py`.

LM Studio must be running with a model loaded. The default model identifier
in `--model` is a placeholder — the user typically passes `--model` or relies
on whatever is currently loaded.

## Conventions to preserve

- **Single file, flat functions.** Don't refactor into a package, classes,
  or multiple modules. The file is small enough to read top-to-bottom.
- **Pydantic models define the LLM contract.** `Track` and `AlbumMetadata`
  are passed as `response_format=` to LM Studio. Changing field names or
  types changes what the model is asked to produce — update the
  `SYSTEM_PROMPT` in lockstep.
- **`SYSTEM_PROMPT` is the spec for extraction.** When you change rules
  (date formats, disambiguator stripping, alignment behavior), edit the
  prompt; don't try to post-process around the model.
- **`--metadata` mode skips the LLM entirely.** It loads
  `<dir>/metadata.json` from a prior run, runs `clean_metadata` as a
  safety net (idempotent), then proceeds straight to `--write`/`--rename`
  and the summary. `load_local_tracks`, `fetch_page_text`, and
  `query_lmstudio` are not called. Keep this path LLM-free.
- **`--url`, `--image`, and `--metadata` are a three-way mutually
  exclusive group; exactly one is required.** Enforced by argparse via
  `add_mutually_exclusive_group(required=True)`. If you add a fourth
  source mode, add it to the same group — don't open-code the
  validation.
- **`IMAGE_SYSTEM_PROMPT` is the spec for image extraction.** Parallel
  to `SYSTEM_PROMPT`. Drops the JSON-LD and wiki-disambiguator rules
  (irrelevant for covers), adds illegible/cropped/obscured guidance.
  Keep both prompts in sync on shared rules (alignment, date format,
  null for missing).
- **`IMAGE_EXTS` is an allowlist of `.jpg/.jpeg/.png/.webp`** — these
  go straight to LM Studio. **`HEIC_EXTS` (`.heic/.heif`) are accepted
  too**, but transparently converted to JPEG via macOS `sips` by the
  `_resolve_image` context manager (cleaned up on exit, including on
  exceptions). HEIC support is macOS-only by design: `sips` not found
  → clean `sys.exit` telling the user to convert externally. Everything
  else (TIFF, BMP, GIF, etc.) is rejected with a clean `sys.exit`. Don't
  introduce Pillow / pyheif / cloud converters without asking.
- **Discogs is special-case.** Their pages block scraping, so
  `DISCOGS_URL_RE` routes to their public JSON API (`fetch_discogs`)
  instead of HTML fetching. Keep that branch when touching `fetch_page_text`.
- **Wikipedia/CMS disambiguators** (`(album)`, `(EP)`, `(song)`, ...) are
  stripped post-hoc in `clean_metadata` *and* discouraged in the prompt.
  Legit parentheticals that are part of the real title must survive — see
  `WIKI_DISAMBIGUATORS` and `strip_disambiguator`.
- **Destructive operations refuse on mismatch.** Both `write_tags` and
  `rename_release` bail with a warning if the file count doesn't equal the
  parsed track count. Preserve that guard.
- **Rename target collisions are refused, not overwritten.** Same for
  per-file renames inside `rename_release`. Don't add an `--overwrite` flag
  without asking.
- **All terminal output goes to stderr via `logging`; the JSON is written
  to a file.** There are two loggers: `logger` (module-level, prefixed
  `INFO:`/`WARNING:`) for progress and warnings, and `summary_logger`
  (`cd_metadata.summary`) for the end-of-run key:value summary, which
  uses a dedicated raw-text handler so its lines render without a level
  prefix. Both write to `sys.stderr`. The parsed JSON is always written
  to disk — `--out` if given, otherwise `<audio_dir>/metadata.json`
  (following `--rename` to the new directory name if used). `print` is
  not used anywhere in `cd_metadata.py`; do not reintroduce it.
- **Color is auto-detected.** `_ColorFormatter` wraps the level prefix
  in ANSI codes when stderr is a TTY and `NO_COLOR` is unset
  (https://no-color.org). The summary block is intentionally uncolored
  so the value text reads cleanly. Don't tint message bodies or summary
  values without asking — that hurts readability of paths.

## Style

- Python 3.10+. `from __future__ import annotations` is used, so PEP 604
  unions (`str | None`) work everywhere.
- Module-level constants in UPPER_SNAKE_CASE at the top of the file.
- No docstrings on small functions; the module docstring carries the
  user-facing description.
- Comments only where the *why* is non-obvious (e.g., the Discogs scraping
  note, the disambiguator explanation). Don't add narration.

## Things not to do without asking

- Don't introduce a config file. CLI flags are the interface.
- Don't add network retry/backoff logic. A single `requests.get` with a
  30s timeout is intentional — the user re-runs on failure.
- Don't switch package manager away from `uv`.
