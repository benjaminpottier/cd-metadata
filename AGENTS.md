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

- `cd_metadata.py` — the core tool. One file, no package.
- `test_cd_metadata.py` — pytest suite.
- `pyproject.toml` — dependencies, managed by `uv`. Test deps live in the `dev` dependency group.
- `uv.lock` — pinned lockfile; commit changes when deps move.
- **Use `uv` for everything**: dependency management, running, testing. Don't forget it's required.
- `command.example` — example invocation.
- `.venv/` — created by `uv sync`. Do not commit.

There is intentionally no `src/` layout and no CI. Tests sit next to the
tool and run with `pytest`.

## Running it

```
uv sync                        # one-time, installs into .venv

# First pass — text mode: LM Studio call, writes <dir>/metadata.json
uv run cd_metadata.py --url URL --dir PATH_TO_RIPS

# Or — image mode: send one or more photos to a vision-capable model
uv run cd_metadata.py --image PATH_TO_IMAGE [PATH_TO_IMAGE ...] --dir PATH_TO_RIPS

# Second pass: reuse the metadata, apply tags/rename without re-calling LM Studio
uv run cd_metadata.py --metadata --dir PATH_TO_RIPS --write --rename
```

Exactly one of `--url`, `--image`, or `--metadata` is required (argparse
mutually exclusive group). `--metadata` reads `<dir>/metadata.json` and
skips the network fetch + LM Studio call entirely. `--image` accepts one
or more JPEG/PNG/WebP (`IMAGE_EXTS`) or HEIC/HEIF (`HEIC_EXTS`) files;
every image is routed through macOS `sips` for normalization (format
convert if needed, EXIF orientation bake, optional downscale to
`IMAGE_MAX_DIMENSION = 2048` on the longest side). Multiple images are
attached to the same chat turn as different views of the same release.

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
- **Pydantic models define the LLM contracts.** `Track` and
  `AlbumMetadata` are the final shape. The URL flow uses two narrower
  intermediate schemas — `AlbumScalars` (album-level fields + notes) and
  `TrackTitlesAndNotes` (a `list[str]` of titles + notes) — one per LLM
  call, then merges into `AlbumMetadata`. The image flow still uses
  `AlbumMetadata` directly. Changing field names or types changes what
  each model is asked to produce; update the corresponding prompt in
  lockstep.
- **`TrackTitlesAndNotes.track_titles` is `list[str]`, not
  `list[Track]` and not `list[str | None]`.** Local quantized models
  under structured output exhibit a strong null bias on `str | None`
  fields — when the schema is `list[Track]` with `Track.title: str | None`,
  every track in the release came back with `title: null` even when the
  page clearly stated all titles (the constrained decoder picks `null`
  as the easier completion, and once one position is null, consistency
  makes the rest null too). The required-string schema removes the null
  escape hatch; the prompt instead tells the model to emit "" for "not
  stated on the page". `query_lmstudio` then maps "" back to `None` and
  builds `Track` objects with the right `track_number`. Don't reintroduce
  `Track | None` in the LLM contract without re-validating extraction
  quality against a weak local model.
- **`album`, `album_artist`, and `Track.title` are deliberately
  Optional.** When they were required, the JSON schema forbade null
  → the model confabulated values it could not actually read (most
  visible with `--image`). With them Optional, the model can return
  null and the user fixes it via the `--metadata` workflow. Do not
  make them required again without understanding this trade-off.
  `write_tags` and `rename_release` refuse to act when these are null
  and tell the user to edit `metadata.json` — preserve that guard.
- **The URL flow is two-stage by design.** `query_lmstudio` makes
  TWO sequential `model.respond` calls: pass 1 fills `AlbumScalars`
  (album/album_artist/date/label/catalog_number/genre + notes), pass 2
  fills `TrackTitlesAndNotes`. Local quantized models under structured output
  default to null on optional string fields when the response schema is
  large; splitting the work in two keeps each call's cognitive load
  light and is what actually makes `album`/`album_artist` populate
  reliably across diverse pages (Discogs JSON, Bandcamp, Wikipedia,
  label sites). The scalars pass deliberately omits the user's track
  list — filenames like "Unknown" would only prime the model toward
  null. The tracks pass receives the scalars from pass 1 as confirmed
  context so the model isn't tempted to second-guess the album identity.
  Notes from both passes are concatenated. Don't collapse this back
  into a single call without re-validating extraction quality.
- **`SCALAR_SYSTEM_PROMPT` and `TRACKS_SYSTEM_PROMPT` are the URL-flow
  specs.** When you change extraction rules (date formats, disambiguator
  stripping, alignment behavior), edit the relevant prompt; don't try
  to post-process around the model. Keep them in sync with
  `IMAGE_SYSTEM_PROMPT` on shared rules.
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
  to the URL-flow prompts. Drops the JSON-LD and wiki-disambiguator
  rules (irrelevant for covers — `clean_metadata` strips disambiguators
  post-hoc regardless), adds illegible/cropped/obscured guidance. Keep
  the prompts in sync on shared rules (alignment, date format, null
  for missing). The prompt's "Core rules" section deliberately
  includes the anti-bias clause: "the user's ripped track list
  (filenames and durations) is for ALIGNMENT only ... do not use
  track count or durations to identify the album." Don't soften that
  — it's what stops the model from guessing the album from track
  count + durations.
- **Image attachment is sandwiched between two text parts.**
  `query_lmstudio_image` takes a `list[Path]` and calls
  `chat.add_user_message([intro, *handles, alignment])` — the lmstudio
  SDK accepts a list mixing strings and `FileHandle`. `build_image_chat_text`
  returns the two text parts (with phrasing that switches singular/plural
  based on `num_images`); image(s) go between them so the model meets
  instructions, then the image(s), then the track-alignment data (not
  before). This structure measurably reduces image-mode hallucinations.
- **Multiple images are supported via `nargs="+"` on `--image`.** They
  are framed to the model as different views of the same release
  (back cover + insert + label, etc.) — the `IMAGE_SYSTEM_PROMPT` tells
  it to combine information across them and treat a field as null only
  if absent/unreadable across all of them. `main` validates every path
  up-front, then uses `contextlib.ExitStack` to manage one `_resolve_image`
  context per file so temp JPEGs are cleaned up even if one image fails.
- **All accepted images route through `sips` via `_resolve_image`.**
  `IMAGE_EXTS` (`.jpg/.jpeg/.png/.webp`) and `HEIC_EXTS` (`.heic/.heif`)
  alike are converted to a JPEG temp file with EXIF orientation baked
  into pixels and the longest dimension capped at `IMAGE_MAX_DIMENSION`
  (2048). The cap is gated by a `sips -g pixelWidth -g pixelHeight`
  pre-check (`_image_needs_resize`) because `sips -Z` is bidirectional
  — without the gate, smaller back-cover photos would be UPSCALED, which
  is exactly the opposite of what we want. If sips can't read the dims
  (returncode != 0 or non-numeric output), `_image_needs_resize` returns
  True so the convert call surfaces the real error rather than silently
  skipping the cap. All sips invocations go through `_run_sips`, which
  centralizes the `FileNotFoundError → clean sys.exit` ("requires macOS")
  handling. Image mode is macOS-only by design; everything else (TIFF,
  BMP, GIF, etc.) is rejected at CLI parse with a clean `sys.exit`.
  Don't introduce Pillow / pyheif / cloud converters without asking.
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
- **`log_summary` emits a `Status:` line that mirrors the
  write/rename guards.** If `album`, `album_artist`, and every
  `Track.title` are non-null, status is "complete — ready to
  --write/--rename"; otherwise "needs review — missing ..." enumerates
  the specific fields. The `Tracks:` line likewise shows "N (all
  titled)" vs. "N (K titled, M missing)". When changing the
  `write_tags`/`rename_release` null guards, update the status check
  in lockstep so the summary stays truthful about what's actionable.
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
