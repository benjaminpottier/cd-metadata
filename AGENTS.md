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
- `pyproject.toml` — dependencies, managed by `uv`.
- `uv.lock` — pinned lockfile; commit changes when deps move.
- `command.example` — example invocation.
- `.venv/` — created by `uv sync`. Do not commit.

There is intentionally no `src/` layout, no tests directory, no CI. Don't add
those unless explicitly asked.

## Running it

```
uv sync                        # one-time, installs into .venv
uv run cd_metadata.py --url URL --dir PATH_TO_RIPS [--write] [--rename] [--out tags.json]
```

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
- **`stderr` for progress, `stdout` for the JSON payload.** The JSON on
  stdout is the machine-readable output; everything else (`Scanning …`,
  `Tagged: …`, warnings) goes to stderr so the JSON stays pipe-clean.

## Style

- Python 3.10+. `from __future__ import annotations` is used, so PEP 604
  unions (`str | None`) work everywhere.
- Module-level constants in UPPER_SNAKE_CASE at the top of the file.
- No docstrings on small functions; the module docstring carries the
  user-facing description.
- Comments only where the *why* is non-obvious (e.g., the Discogs scraping
  note, the disambiguator explanation). Don't add narration.

## Things not to do without asking

- Don't add a logging framework. `print(..., file=sys.stderr)` is the
  convention here.
- Don't introduce a config file. CLI flags are the interface.
- Don't add network retry/backoff logic. A single `requests.get` with a
  30s timeout is intentional — the user re-runs on failure.
- Don't add tests scaffolding (`pytest`, `tox`, etc.) unless asked.
- Don't switch package manager away from `uv`.
