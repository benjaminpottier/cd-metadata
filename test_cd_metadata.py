"""Tests for cd_metadata.py — aims for 100% line and branch coverage.

Externals (requests, lmstudio, mutagen) are mocked so tests run hermetically.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import cd_metadata as cdm
from cd_metadata import (
    AlbumMetadata,
    AlbumScalars,
    Track,
    TrackTitlesAndNotes,
    build_image_chat_text,
    build_scalar_message,
    build_tracks_message,
    clean_metadata,
    fetch_discogs,
    fetch_page_text,
    load_local_tracks,
    main,
    query_lmstudio,
    query_lmstudio_image,
    rename_release,
    sanitize,
    strip_disambiguator,
    write_tags,
)


# ---------- helpers ----------

def make_meta(tracks=None, **kw):
    defaults = {
        "album": "Some Album",
        "album_artist": "Some Artist",
        "tracks": tracks if tracks is not None else [
            Track(track_number=1, title="One"),
            Track(track_number=2, title="Two"),
        ],
    }
    defaults.update(kw)
    return AlbumMetadata(**defaults)


def touch(path: Path, content: bytes = b"") -> Path:
    path.write_bytes(content)
    return path


class FakeResponse:
    def __init__(self, *, text="", json_data=None, status=200):
        self.text = text
        self._json = json_data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class FakeAudio:
    """Dict-like stand-in for mutagen.File(..., easy=True)."""

    def __init__(self):
        self.data: dict[str, str] = {}
        self.saved = False

    def __setitem__(self, key, value):
        self.data[key] = value

    def save(self):
        self.saved = True


# ---------- strip_disambiguator / clean_metadata ----------

class TestStripDisambiguator:
    def test_strips_album_suffix(self):
        assert strip_disambiguator("Discovery (album)") == "Discovery"

    def test_strips_case_insensitive(self):
        assert strip_disambiguator("Foo (ALBUM)") == "Foo"

    def test_preserves_legit_parenthetical(self):
        # Not in WIKI_DISAMBIGUATORS — must survive.
        title = "Smells Like Teen Spirit (Single Edit)"
        assert strip_disambiguator(title) == title

    def test_no_parenthetical(self):
        assert strip_disambiguator("Plain Title") == "Plain Title"


def test_clean_metadata_strips_album_and_tracks():
    meta = AlbumMetadata(
        album="Discovery (album)",
        album_artist="Daft Punk",
        tracks=[
            Track(track_number=1, title="One More Time (song)"),
            Track(track_number=2, title="Aerodynamic"),
        ],
    )
    cleaned = clean_metadata(meta)
    assert cleaned.album == "Discovery"
    assert cleaned.tracks[0].title == "One More Time"
    assert cleaned.tracks[1].title == "Aerodynamic"


def test_clean_metadata_handles_none_fields():
    """Cover the False side of `if meta.album is not None` and `if t.title is not None`."""
    meta = AlbumMetadata(
        album=None,
        album_artist=None,
        tracks=[
            Track(track_number=1, title=None),
            Track(track_number=2, title="Visible (album)"),  # this one still strips
        ],
    )
    cleaned = clean_metadata(meta)
    assert cleaned.album is None
    assert cleaned.tracks[0].title is None
    assert cleaned.tracks[1].title == "Visible"


# ---------- sanitize ----------

class TestSanitize:
    def test_replaces_invalid_chars(self):
        assert sanitize("foo/bar:baz") == "foo - bar - baz"

    def test_strips_trailing_dots_spaces(self):
        assert sanitize("Hello.  ") == "Hello"

    def test_empty_after_strip_returns_untitled(self):
        assert sanitize("...") == "Untitled"

    def test_passthrough(self):
        assert sanitize("Normal Name") == "Normal Name"


# ---------- build_scalar_message ----------

class TestBuildScalarMessage:
    def test_omits_filenames(self):
        # The scalars pass intentionally does not include the user's track
        # list — filenames like "Unknown" would only prime the model toward
        # null. Album-level fields don't need alignment data anyway.
        msg = build_scalar_message("http://x", "page body")
        assert "http://x" in msg
        assert "page body" in msg
        assert "Unknown" not in msg
        assert "ripped track" not in msg

    def test_truncation(self):
        long_page = "Z" * (cdm.PAGE_CHAR_BUDGET + 100)
        msg = build_scalar_message("http://example", long_page)
        assert "100 more chars truncated" in msg
        assert msg.count("Z") == cdm.PAGE_CHAR_BUDGET


# ---------- build_tracks_message ----------

class TestBuildTracksMessage:
    def test_includes_filenames_after_page(self):
        msg = build_tracks_message(
            "http://x",
            "page body",
            [{"position": 1, "filename": "01.flac", "duration_seconds": 30.5}],
            AlbumScalars(),
        )
        assert "http://x" in msg
        assert "01.flac" in msg
        assert "30.5" in msg
        assert "1 ripped track(s)" in msg
        assert "page body" in msg
        assert "truncated" not in msg
        # Page text must precede the filename block.
        assert msg.index("page body") < msg.index("01.flac")

    def test_truncation(self):
        long_page = "Z" * (cdm.PAGE_CHAR_BUDGET + 100)
        msg = build_tracks_message(
            "http://example", long_page, [], AlbumScalars(),
        )
        assert "100 more chars truncated" in msg
        assert msg.count("Z") == cdm.PAGE_CHAR_BUDGET

    def test_includes_scalar_context_block_when_present(self):
        msg = build_tracks_message(
            "http://x",
            "page",
            [],
            AlbumScalars(
                album="Atmosphere For Lovers And Thieves",
                album_artist="Ben Webster",
                date="1990",
            ),
        )
        assert "Album context" in msg
        assert "album: Atmosphere For Lovers And Thieves" in msg
        assert "album_artist: Ben Webster" in msg
        assert "date: 1990" in msg

    def test_omits_context_block_when_scalars_empty(self):
        msg = build_tracks_message("http://x", "page", [], AlbumScalars())
        assert "Album context" not in msg


# ---------- load_local_tracks ----------

def test_load_local_tracks_empty_dir(tmp_path):
    with pytest.raises(SystemExit):
        load_local_tracks(tmp_path)


def test_load_local_tracks_ignores_non_audio(tmp_path):
    touch(tmp_path / "cover.jpg")
    with pytest.raises(SystemExit):
        load_local_tracks(tmp_path)


def test_load_local_tracks_with_durations(tmp_path, monkeypatch):
    touch(tmp_path / "02.flac")
    touch(tmp_path / "01.flac")

    def fake_file(p):
        m = MagicMock()
        m.info.length = 123.456
        return m

    monkeypatch.setattr(cdm, "mutagen", MagicMock(File=fake_file))
    rows = load_local_tracks(tmp_path)
    assert [r["filename"] for r in rows] == ["01.flac", "02.flac"]
    assert all(r["duration_seconds"] == 123.46 for r in rows)
    assert [r["position"] for r in rows] == [1, 2]


def test_load_local_tracks_handles_none_mutagen(tmp_path, monkeypatch):
    touch(tmp_path / "track.mp3")
    monkeypatch.setattr(cdm, "mutagen", MagicMock(File=lambda p: None))
    rows = load_local_tracks(tmp_path)
    assert rows[0]["duration_seconds"] is None


def test_load_local_tracks_handles_no_info(tmp_path, monkeypatch):
    touch(tmp_path / "track.mp3")
    m = MagicMock()
    m.info = None
    monkeypatch.setattr(cdm, "mutagen", MagicMock(File=lambda p: m))
    rows = load_local_tracks(tmp_path)
    assert rows[0]["duration_seconds"] is None


# ---------- fetch_discogs ----------

class TestFetchDiscogs:
    def test_releases(self, monkeypatch):
        captured = {}

        def fake_get(url, timeout, headers):
            captured["url"] = url
            captured["headers"] = headers
            return FakeResponse(json_data={
                "title": "T", "year": 2001, "x_drop_me": 1, "labels": [],
            })

        monkeypatch.setattr(cdm.requests, "get", fake_get)
        out = fetch_discogs("release", "42")
        assert "releases/42" in out
        assert "x_drop_me" not in out
        assert "title" in out
        assert captured["url"].endswith("/releases/42")
        assert captured["headers"]["User-Agent"] == cdm.USER_AGENT

    def test_masters_via_master(self, monkeypatch):
        monkeypatch.setattr(
            cdm.requests, "get",
            lambda *a, **k: FakeResponse(json_data={"title": "M"}),
        )
        out = fetch_discogs("master", "9")
        assert "masters/9" in out

    def test_masters_via_m_uppercase(self, monkeypatch):
        monkeypatch.setattr(
            cdm.requests, "get",
            lambda *a, **k: FakeResponse(json_data={"title": "M"}),
        )
        out = fetch_discogs("M", "7")
        assert "masters/7" in out


# ---------- fetch_page_text ----------

class TestFetchPageText:
    def test_discogs_url_dispatches(self, monkeypatch):
        monkeypatch.setattr(
            cdm, "fetch_discogs",
            lambda kind, ident: f"DISCOGS:{kind}:{ident}",
        )
        out = fetch_page_text("https://www.discogs.com/release/12345")
        assert out == "DISCOGS:release:12345"

    def test_html_with_ld_json(self, monkeypatch):
        html = (
            '<html><head>'
            '<script type="application/ld+json">{"@type":"MusicAlbum"}</script>'
            '<script type="application/ld+json"></script>'
            '</head>'
            '<body><nav>NAVTEXT</nav><p>BodyTitle</p>'
            '<script>var x=1</script></body></html>'
        )
        monkeypatch.setattr(
            cdm.requests, "get",
            lambda *a, **k: FakeResponse(text=html),
        )
        out = fetch_page_text("https://example.com/album")
        assert "JSON-LD" in out
        assert "MusicAlbum" in out
        assert "PAGE TEXT" in out
        assert "BodyTitle" in out
        assert "var x=1" not in out  # script tag stripped
        assert "NAVTEXT" not in out.split("PAGE TEXT", 1)[1]

    def test_html_without_ld_json(self, monkeypatch):
        html = "<html><body><p>Just body text</p></body></html>"
        monkeypatch.setattr(
            cdm.requests, "get",
            lambda *a, **k: FakeResponse(text=html),
        )
        out = fetch_page_text("https://example.com/album")
        assert "JSON-LD" not in out
        assert "Just body text" in out


# ---------- query_lmstudio ----------

class TestQueryLmstudio:
    @staticmethod
    def _setup(monkeypatch, scalars, tracks_and_notes):
        """Wire up `lms.Client` so two sequential `model.respond` calls return
        the scalars and the tracks payload respectively.
        """
        fake_lms = MagicMock()
        client = fake_lms.Client.return_value.__enter__.return_value
        fake_lms.Client.return_value.__exit__.return_value = False
        model = client.llm.model.return_value
        scalars_result = MagicMock()
        scalars_result.parsed = scalars
        tracks_result = MagicMock()
        tracks_result.parsed = tracks_and_notes
        model.respond.side_effect = [scalars_result, tracks_result]
        monkeypatch.setattr(cdm, "lms", fake_lms)
        return fake_lms, client, model

    def test_with_model_name_and_pydantic_parsed(self, monkeypatch):
        scalars = AlbumScalars(album="A", album_artist="B")
        tan = TrackTitlesAndNotes(track_titles=["T"])
        _, client, model = self._setup(monkeypatch, scalars, tan)
        out = query_lmstudio("my-model", "http://x", "page", [])
        assert isinstance(out, AlbumMetadata)
        assert out.album == "A"
        assert out.album_artist == "B"
        assert out.tracks[0].track_number == 1
        assert out.tracks[0].title == "T"
        client.llm.model.assert_called_once_with("my-model")
        # Two passes: scalars then tracks.
        assert model.respond.call_count == 2

    def test_without_model_name(self, monkeypatch):
        scalars = AlbumScalars()
        tan = TrackTitlesAndNotes(track_titles=[])
        _, client, _ = self._setup(monkeypatch, scalars, tan)
        query_lmstudio(None, "http://x", "page", [])
        client.llm.model.assert_called_once_with()

    def test_dict_parsed_uses_model_validate(self, monkeypatch):
        # Both passes return raw dicts (some lms SDK versions skip Pydantic
        # parsing). query_lmstudio must validate them into the right models.
        scalars_dict = {"album": "A (album)", "album_artist": "B"}
        tracks_dict = {"track_titles": ["T (song)"]}
        self._setup(monkeypatch, scalars_dict, tracks_dict)
        out = query_lmstudio(None, "http://x", "page", [])
        # clean_metadata strips disambiguators.
        assert out.album == "A"
        assert out.tracks[0].title == "T"

    def test_empty_string_title_maps_to_none(self, monkeypatch):
        # The "" sentinel from the model means "not stated on the page".
        # Downstream guards (write_tags, rename_release) check `is None`,
        # so we MUST convert "" → None here.
        scalars = AlbumScalars()
        tan = TrackTitlesAndNotes(track_titles=["", "Real Title", "   "])
        self._setup(monkeypatch, scalars, tan)
        out = query_lmstudio(None, "http://x", "page", [])
        assert [t.title for t in out.tracks] == [None, "Real Title", None]
        assert [t.track_number for t in out.tracks] == [1, 2, 3]

    def test_padding_when_model_returns_fewer_titles(self, monkeypatch):
        # Local model emitted only 2 titles but the rip has 3 files. Pad
        # with title=None so positions stay 1:1 with the audio.
        scalars = AlbumScalars()
        tan = TrackTitlesAndNotes(track_titles=["A", "B"])
        self._setup(monkeypatch, scalars, tan)
        local = [
            {"position": i, "filename": f"{i}.flac", "duration_seconds": 1.0}
            for i in (1, 2, 3)
        ]
        out = query_lmstudio(None, "http://x", "page", local)
        assert [t.title for t in out.tracks] == ["A", "B", None]

    def test_truncation_when_model_returns_more_titles(self, monkeypatch):
        # Local model overran (4 titles, 2 ripped files). Truncate so the
        # tracks list matches the audio count — write_tags refuses on a
        # mismatch and we don't want stray phantom tracks.
        scalars = AlbumScalars()
        tan = TrackTitlesAndNotes(track_titles=["A", "B", "C", "D"])
        self._setup(monkeypatch, scalars, tan)
        local = [
            {"position": i, "filename": f"{i}.flac", "duration_seconds": 1.0}
            for i in (1, 2)
        ]
        out = query_lmstudio(None, "http://x", "page", local)
        assert [t.title for t in out.tracks] == ["A", "B"]

    def test_no_local_tracks_uses_model_count(self, monkeypatch):
        # If the caller supplies an empty local_tracks list, fall back to
        # the model's emitted length rather than zeroing everything out.
        scalars = AlbumScalars()
        tan = TrackTitlesAndNotes(track_titles=["A", "B"])
        self._setup(monkeypatch, scalars, tan)
        out = query_lmstudio(None, "http://x", "page", [])
        assert [t.title for t in out.tracks] == ["A", "B"]

    def test_notes_from_both_passes_merge(self, monkeypatch):
        scalars = AlbumScalars(album="A", notes="album title not visible")
        tan = TrackTitlesAndNotes(
            track_titles=["T"],
            notes="track 5 duration off by 8s",
        )
        self._setup(monkeypatch, scalars, tan)
        out = query_lmstudio(None, "http://x", "page", [])
        assert out.notes == "album title not visible\ntrack 5 duration off by 8s"

    def test_notes_none_when_both_passes_silent(self, monkeypatch):
        scalars = AlbumScalars()
        tan = TrackTitlesAndNotes(track_titles=[])
        self._setup(monkeypatch, scalars, tan)
        out = query_lmstudio(None, "http://x", "page", [])
        assert out.notes is None

    def test_notes_only_from_one_pass(self, monkeypatch):
        scalars = AlbumScalars(notes="only scalar notes")
        tan = TrackTitlesAndNotes(track_titles=[])
        self._setup(monkeypatch, scalars, tan)
        out = query_lmstudio(None, "http://x", "page", [])
        assert out.notes == "only scalar notes"


# ---------- build_image_chat_text ----------

class TestBuildImageChatText:
    def test_returns_intro_and_alignment_parts(self):
        tracks = [
            {"position": 1, "filename": "01.flac", "duration_seconds": 30.5},
            {"position": 2, "filename": "02.flac", "duration_seconds": 45.0},
        ]
        intro, alignment = build_image_chat_text(tracks)
        # Intro must tell the model to transcribe from the image and warn
        # against using the track list as identification.
        assert "from the image" in intro.lower()
        assert "alignment" in intro.lower()
        assert "2 ripped track(s)" in intro
        # Singular phrasing when only one image.
        assert "Below is an image" in intro
        # Alignment carries the actual file list AFTER the image.
        assert "01.flac" in alignment
        assert "02.flac" in alignment
        # No URL/page-text leakage from the text-mode helper.
        assert "Source URL" not in intro and "Source URL" not in alignment
        assert "PAGE TEXT" not in intro and "PAGE TEXT" not in alignment

    def test_plural_phrasing_for_multiple_images(self):
        tracks = [
            {"position": 1, "filename": "01.flac", "duration_seconds": 30.5},
        ]
        intro, alignment = build_image_chat_text(tracks, num_images=3)
        # Intro signals multiple images and combining-views guidance.
        assert "3 images" in intro
        assert "across the images" in intro
        assert "any image" in intro
        # Alignment text references the multi-image read.
        assert "from the images" in alignment


# ---------- query_lmstudio_image ----------

class TestQueryLmstudioImage:
    @staticmethod
    def _setup(monkeypatch, parsed, num_handles=1):
        fake_lms = MagicMock()
        client = fake_lms.Client.return_value.__enter__.return_value
        fake_lms.Client.return_value.__exit__.return_value = False
        handles = [MagicMock(name=f"FileHandle{i}") for i in range(num_handles)]
        client.prepare_image.side_effect = handles
        model = client.llm.model.return_value
        model.respond.return_value.parsed = parsed
        monkeypatch.setattr(cdm, "lms", fake_lms)
        return fake_lms, client, model, handles

    def test_passes_image_handle_to_chat(self, tmp_path, monkeypatch):
        img = tmp_path / "back.jpg"
        img.write_bytes(b"")
        meta = make_meta()
        fake_lms, client, _, handles = self._setup(monkeypatch, meta)
        out = query_lmstudio_image("my-model", [img], [])
        assert isinstance(out, AlbumMetadata)
        client.prepare_image.assert_called_once_with(img)
        client.llm.model.assert_called_once_with("my-model")
        # Content is a [intro, handle, alignment] list — image sandwiched
        # between text parts.
        chat = fake_lms.Chat.return_value
        args, _ = chat.add_user_message.call_args
        content = args[0]
        assert isinstance(content, list) and len(content) == 3
        assert isinstance(content[0], str)
        assert content[1] is handles[0]
        assert isinstance(content[2], str)
        # And the chat is built from the image-specific system prompt.
        fake_lms.Chat.assert_called_once_with(cdm.IMAGE_SYSTEM_PROMPT)

    def test_multiple_images_attached_in_order(self, tmp_path, monkeypatch):
        a = tmp_path / "back.jpg"
        b = tmp_path / "insert.jpg"
        a.write_bytes(b"")
        b.write_bytes(b"")
        fake_lms, client, _, handles = self._setup(
            monkeypatch, make_meta(), num_handles=2,
        )
        query_lmstudio_image("my-model", [a, b], [])
        # Both images prepared, in argument order.
        assert [c.args[0] for c in client.prepare_image.call_args_list] == [a, b]
        chat = fake_lms.Chat.return_value
        args, _ = chat.add_user_message.call_args
        content = args[0]
        # Content is [intro, handle_a, handle_b, alignment].
        assert len(content) == 4
        assert isinstance(content[0], str)
        assert content[1] is handles[0]
        assert content[2] is handles[1]
        assert isinstance(content[3], str)

    def test_without_model_name(self, tmp_path, monkeypatch):
        img = tmp_path / "back.png"
        img.write_bytes(b"")
        _, client, _, _ = self._setup(monkeypatch, make_meta())
        query_lmstudio_image(None, [img], [])
        client.llm.model.assert_called_once_with()

    def test_dict_parsed_uses_model_validate(self, tmp_path, monkeypatch):
        img = tmp_path / "back.png"
        img.write_bytes(b"")
        parsed_dict = {
            "album": "A (album)",
            "album_artist": "B",
            "tracks": [{"track_number": 1, "title": "T (song)"}],
        }
        self._setup(monkeypatch, parsed_dict)
        out = query_lmstudio_image(None, [img], [])
        assert out.album == "A"
        assert out.tracks[0].title == "T"


# ---------- write_tags ----------

class TestWriteTags:
    def test_refuses_on_mismatch(self, tmp_path, caplog, monkeypatch):
        caplog.set_level(logging.WARNING, logger="cd_metadata")
        touch(tmp_path / "01.flac")
        meta = make_meta()  # 2 tracks vs 1 file
        called = []
        monkeypatch.setattr(
            cdm, "mutagen",
            MagicMock(File=lambda *a, **k: called.append(1)),
        )
        write_tags(tmp_path, meta)
        assert "refusing to write" in caplog.text
        assert called == []

    def test_happy_path_with_all_optionals(self, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger="cd_metadata")
        touch(tmp_path / "01.flac")
        touch(tmp_path / "02.flac")
        meta = make_meta(
            date="1999", genre="Rock",
            tracks=[
                Track(track_number=1, title="A", artist="X", composer="C1"),
                Track(track_number=2, title="B"),
            ],
        )
        objs = {p.name: FakeAudio() for p in tmp_path.iterdir()}
        monkeypatch.setattr(
            cdm, "mutagen",
            MagicMock(File=lambda path, easy=False: objs[path.name]),
        )
        write_tags(tmp_path, meta)
        assert "Tagged: 01.flac" in caplog.text
        assert "Tagged: 02.flac" in caplog.text
        first = objs["01.flac"].data
        assert first["title"] == "A"
        assert first["artist"] == "X"
        assert first["composer"] == "C1"
        assert first["date"] == "1999"
        assert first["genre"] == "Rock"
        assert objs["01.flac"].saved
        second = objs["02.flac"].data
        assert second["artist"] == "Some Artist"  # fell back to album_artist
        assert "composer" not in second

    def test_happy_path_without_optionals(self, tmp_path, monkeypatch):
        """Cover the False side of `if meta.date`, `if meta.genre`, `if composer`."""
        touch(tmp_path / "01.flac")
        meta = make_meta(tracks=[Track(track_number=1, title="Only")])
        obj = FakeAudio()
        monkeypatch.setattr(
            cdm, "mutagen",
            MagicMock(File=lambda *a, **k: obj),
        )
        write_tags(tmp_path, meta)
        assert "date" not in obj.data
        assert "genre" not in obj.data
        assert "composer" not in obj.data

    def test_skips_unsupported_format(self, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.WARNING, logger="cd_metadata")
        touch(tmp_path / "01.flac")
        meta = make_meta(tracks=[Track(track_number=1, title="A")])
        monkeypatch.setattr(
            cdm, "mutagen",
            MagicMock(File=lambda *a, **k: None),
        )
        write_tags(tmp_path, meta)
        assert "unsupported tag format" in caplog.text

    def test_refuses_on_null_album(self, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.WARNING, logger="cd_metadata")
        touch(tmp_path / "01.flac")
        meta = make_meta(
            album=None,
            tracks=[Track(track_number=1, title="A")],
        )
        called: list = []
        monkeypatch.setattr(
            cdm, "mutagen",
            MagicMock(File=lambda *a, **k: called.append(1)),
        )
        write_tags(tmp_path, meta)
        assert "null" in caplog.text.lower()
        assert "refusing to write" in caplog.text
        assert called == []

    def test_refuses_on_null_track_title(self, tmp_path, monkeypatch, caplog):
        caplog.set_level(logging.WARNING, logger="cd_metadata")
        touch(tmp_path / "01.flac")
        meta = make_meta(tracks=[Track(track_number=1, title=None)])
        called: list = []
        monkeypatch.setattr(
            cdm, "mutagen",
            MagicMock(File=lambda *a, **k: called.append(1)),
        )
        write_tags(tmp_path, meta)
        assert "null" in caplog.text.lower()
        assert called == []


# ---------- rename_release ----------

class TestRenameRelease:
    def test_refuses_on_mismatch(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="cd_metadata")
        d = tmp_path / "rips"
        d.mkdir()
        touch(d / "a.flac")  # 1 file
        meta = make_meta()  # 2 tracks
        assert rename_release(d, meta) is None
        assert "refusing to rename" in caplog.text

    def test_happy_path(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="cd_metadata")
        d = tmp_path / "rips"
        d.mkdir()
        touch(d / "a.flac")
        touch(d / "b.flac")
        meta = make_meta(date="2020-01-01")
        result = rename_release(d, meta)
        assert result is not None
        assert result.name == "Some Artist - Some Album (2020) [FLAC]"
        names = sorted(p.name for p in result.iterdir())
        assert names == ["01. One.flac", "02. Two.flac"]
        assert "Renamed dir" in caplog.text
        assert "Renamed:" in caplog.text

    def test_target_dir_exists_collision(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="cd_metadata")
        d = tmp_path / "rips"
        d.mkdir()
        touch(d / "a.flac")
        touch(d / "b.flac")
        (tmp_path / "Some Artist - Some Album (Unknown) [FLAC]").mkdir()
        assert rename_release(d, make_meta()) is None
        assert "target directory already exists" in caplog.text

    def test_audio_dir_already_correct(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="cd_metadata")
        d = tmp_path / "Some Artist - Some Album (Unknown) [FLAC]"
        d.mkdir()
        touch(d / "01. One.flac")  # already-correct name → continue branch
        touch(d / "b.flac")
        meta = make_meta(tracks=[
            Track(track_number=1, title="One"),
            Track(track_number=2, title="Two"),
        ])
        result = rename_release(d, meta)
        assert result == d
        assert "Renamed dir" not in caplog.text  # dir name already matched
        assert "b.flac -> 02. Two.flac" in caplog.text
        assert sorted(p.name for p in d.iterdir()) == [
            "01. One.flac", "02. Two.flac",
        ]

    def test_per_file_collision(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="cd_metadata")
        d = tmp_path / "rips"
        d.mkdir()
        touch(d / "aaa.flac")
        touch(d / "bbb.flac")
        # Two tracks share track_number+title → both target the same path.
        meta = make_meta(tracks=[
            Track(track_number=1, title="Same"),
            Track(track_number=1, title="Same"),
        ])
        rename_release(d, meta)
        assert "target exists, skipping" in caplog.text

    def test_refuses_on_null_album_artist(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="cd_metadata")
        d = tmp_path / "rips"
        d.mkdir()
        touch(d / "a.flac")
        touch(d / "b.flac")
        meta = make_meta(album_artist=None)  # default tracks have titles
        assert rename_release(d, meta) is None
        assert "null" in caplog.text.lower()
        assert "refusing to rename" in caplog.text
        # Directory not renamed.
        assert d.exists()

    def test_refuses_on_null_track_title(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING, logger="cd_metadata")
        d = tmp_path / "rips"
        d.mkdir()
        touch(d / "a.flac")
        touch(d / "b.flac")
        meta = make_meta(tracks=[
            Track(track_number=1, title="Visible"),
            Track(track_number=2, title=None),
        ])
        assert rename_release(d, meta) is None
        assert "null" in caplog.text.lower()


# ---------- main ----------

# ---------- color formatter ----------

class TestColorFormatter:
    def _record(self, level):
        return logging.LogRecord(
            "x", level, "x", 0, "hello", None, None,
        )

    def test_colors_levelname_when_enabled(self):
        fmt = cdm._ColorFormatter("%(levelname)s: %(message)s", color=True)
        out = fmt.format(self._record(logging.WARNING))
        # Yellow escape on, reset after the level name, message uncolored.
        assert out.startswith("\033[33mWARNING\033[0m: hello")

    def test_info_uses_cyan(self):
        fmt = cdm._ColorFormatter("%(levelname)s: %(message)s", color=True)
        out = fmt.format(self._record(logging.INFO))
        assert out.startswith("\033[36mINFO\033[0m: ")

    def test_no_color_when_disabled(self):
        fmt = cdm._ColorFormatter("%(levelname)s: %(message)s", color=False)
        out = fmt.format(self._record(logging.WARNING))
        assert "\033[" not in out
        assert out == "WARNING: hello"

    def test_unknown_level_skips_color(self):
        """Custom level numbers not in the map → no color even when enabled."""
        fmt = cdm._ColorFormatter("%(levelname)s: %(message)s", color=True)
        rec = logging.LogRecord("x", 25, "x", 0, "hello", None, None)
        out = fmt.format(rec)
        assert "\033[" not in out


class _FakeStderr:
    """Stand-in for sys.stderr with a controllable isatty()."""

    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty

    def write(self, s):  # pragma: no cover - unused in these tests
        pass

    def flush(self):  # pragma: no cover - unused in these tests
        pass


def test_configure_logging_attaches_root_handler(monkeypatch):
    """Cover the root.handlers-empty branch of _configure_logging.

    Pytest's caplog plugin pre-attaches handlers to root, so the branch only
    fires when we explicitly clear them.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    root.handlers.clear()
    try:
        monkeypatch.setenv("NO_COLOR", "1")  # decouple from TTY state
        cdm._configure_logging()
        assert len(root.handlers) == 1
        formatter = root.handlers[0].formatter
        assert isinstance(formatter, cdm._ColorFormatter)
        assert formatter.color is False
        assert root.level == logging.INFO
    finally:
        # Restore so caplog keeps working for the rest of the suite.
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)


class TestColorEnabled:
    def test_on_when_tty_and_no_env(self, monkeypatch):
        monkeypatch.setattr(sys, "stderr", _FakeStderr(True))
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert cdm._color_enabled() is True

    def test_off_when_no_color_env(self, monkeypatch):
        monkeypatch.setattr(sys, "stderr", _FakeStderr(True))
        monkeypatch.setenv("NO_COLOR", "1")
        assert cdm._color_enabled() is False

    def test_off_when_not_tty(self, monkeypatch):
        monkeypatch.setattr(sys, "stderr", _FakeStderr(False))
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert cdm._color_enabled() is False


def test_main_invalid_dir(tmp_path, monkeypatch):
    bad = tmp_path / "does-not-exist"
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--url", "http://x", "--dir", str(bad),
    ])
    with pytest.raises(SystemExit):
        main()


def test_main_minimal_writes_default_path(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "rips"
    audio.mkdir()
    meta = make_meta(date="2020", label="Some Label")
    monkeypatch.setattr(cdm, "load_local_tracks", lambda d: [])
    monkeypatch.setattr(cdm, "fetch_page_text", lambda url: "PAGE")
    monkeypatch.setattr(cdm, "query_lmstudio", lambda *a, **k: meta)
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--url", "http://x", "--dir", str(audio),
    ])
    main()
    default_out = audio / "metadata.json"
    assert default_out.exists()
    assert json.loads(default_out.read_text())["album"] == "Some Album"
    # Nothing on stdout — JSON lives in the file.
    captured = capsys.readouterr()
    assert captured.out == ""
    # Summary on stderr.
    assert "Album:        Some Album" in captured.err
    assert "Album artist: Some Artist" in captured.err
    assert "Year:         2020" in captured.err
    assert "Label:        Some Label" in captured.err
    assert "Tracks:       2" in captured.err
    assert f"Wrote:        {default_out}" in captured.err


def test_main_summary_handles_missing_optional_fields(tmp_path, monkeypatch, capsys):
    """date and label are optional → summary renders '-'."""
    audio = tmp_path / "rips"
    audio.mkdir()
    meta = make_meta()  # no date, no label, no notes
    monkeypatch.setattr(cdm, "load_local_tracks", lambda d: [])
    monkeypatch.setattr(cdm, "fetch_page_text", lambda url: "PAGE")
    monkeypatch.setattr(cdm, "query_lmstudio", lambda *a, **k: meta)
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--url", "http://x", "--dir", str(audio),
    ])
    main()
    err = capsys.readouterr().err
    assert "Year:         -" in err
    assert "Label:        -" in err
    # Notes section is suppressed when meta.notes is empty.
    assert "Notes:" not in err


def test_main_summary_renders_dash_for_null_album(tmp_path, monkeypatch, capsys):
    """Null album / album_artist render as '-' in the summary so the user
    sees they need editing before running --write/--rename."""
    audio = tmp_path / "rips"
    audio.mkdir()
    meta = make_meta(album=None, album_artist=None)
    monkeypatch.setattr(cdm, "load_local_tracks", lambda d: [])
    monkeypatch.setattr(cdm, "fetch_page_text", lambda url: "PAGE")
    monkeypatch.setattr(cdm, "query_lmstudio", lambda *a, **k: meta)
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--url", "http://x", "--dir", str(audio),
    ])
    main()
    err = capsys.readouterr().err
    assert "Album:        -" in err
    assert "Album artist: -" in err


def test_main_metadata_roundtrips_null_fields(tmp_path, monkeypatch, capsys):
    """JSON with null album/artist/title loads, cleans, and re-writes intact."""
    audio = tmp_path / "rips"
    audio.mkdir()
    payload = {
        "album": None,
        "album_artist": None,
        "tracks": [
            {"track_number": 1, "title": None},
            {"track_number": 2, "title": "Legible"},
        ],
        "notes": "back cover photographed at an angle; title obscured.",
    }
    (audio / "metadata.json").write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--metadata", "--dir", str(audio),
    ])
    main()
    on_disk = json.loads((audio / "metadata.json").read_text())
    assert on_disk["album"] is None
    assert on_disk["album_artist"] is None
    assert on_disk["tracks"][0]["title"] is None
    assert on_disk["tracks"][1]["title"] == "Legible"


def test_main_summary_emits_notes_when_present(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "rips"
    audio.mkdir()
    meta = make_meta(notes="Page lists 13 tracks; aligned first 12.\nTrack 5 duration off by 8s.")
    monkeypatch.setattr(cdm, "load_local_tracks", lambda d: [])
    monkeypatch.setattr(cdm, "fetch_page_text", lambda url: "PAGE")
    monkeypatch.setattr(cdm, "query_lmstudio", lambda *a, **k: meta)
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--url", "http://x", "--dir", str(audio),
    ])
    main()
    err = capsys.readouterr().err
    assert "Notes:" in err
    assert "  Page lists 13 tracks; aligned first 12." in err
    assert "  Track 5 duration off by 8s." in err


# ---------- log_summary status line ----------

class TestSummaryStatus:
    """Status reflects whether the extraction is ready for --write/--rename.

    The check mirrors the guards in write_tags/rename_release: album,
    album_artist, and every track title must be non-null. Anything else
    (date, label, genre, catalog_number) is informational only.
    """

    def _run(self, monkeypatch, capsys, meta, tmp_path):
        audio = tmp_path / "rips"
        audio.mkdir()
        monkeypatch.setattr(cdm, "load_local_tracks", lambda d: [])
        monkeypatch.setattr(cdm, "fetch_page_text", lambda url: "PAGE")
        monkeypatch.setattr(cdm, "query_lmstudio", lambda *a, **k: meta)
        monkeypatch.setattr(sys, "argv", [
            "cd_metadata.py", "--url", "http://x", "--dir", str(audio),
        ])
        main()
        return capsys.readouterr().err

    def test_complete_when_all_required_fields_populated(
        self, tmp_path, monkeypatch, capsys,
    ):
        meta = make_meta()  # 2 tracks, both titled
        err = self._run(monkeypatch, capsys, meta, tmp_path)
        assert "Status:       complete — ready to --write/--rename" in err
        assert "Tracks:       2 (all titled)" in err

    def test_needs_review_when_album_null(self, tmp_path, monkeypatch, capsys):
        meta = make_meta(album=None)
        err = self._run(monkeypatch, capsys, meta, tmp_path)
        assert "Status:       needs review — missing album" in err

    def test_needs_review_when_artist_null(self, tmp_path, monkeypatch, capsys):
        meta = make_meta(album_artist=None)
        err = self._run(monkeypatch, capsys, meta, tmp_path)
        assert "Status:       needs review — missing album_artist" in err

    def test_singular_track_title_missing(self, tmp_path, monkeypatch, capsys):
        meta = make_meta(tracks=[
            Track(track_number=1, title="A"),
            Track(track_number=2, title=None),
        ])
        err = self._run(monkeypatch, capsys, meta, tmp_path)
        assert "Tracks:       2 (1 titled, 1 missing)" in err
        assert "missing 1 track title" in err
        # Singular: not "1 track titles".
        assert "1 track titles" not in err

    def test_plural_track_titles_missing(self, tmp_path, monkeypatch, capsys):
        meta = make_meta(tracks=[
            Track(track_number=1, title=None),
            Track(track_number=2, title=None),
            Track(track_number=3, title="C"),
        ])
        err = self._run(monkeypatch, capsys, meta, tmp_path)
        assert "Tracks:       3 (1 titled, 2 missing)" in err
        assert "missing 2 track titles" in err

    def test_multiple_missing_fields_listed_in_order(
        self, tmp_path, monkeypatch, capsys,
    ):
        meta = make_meta(
            album=None,
            album_artist=None,
            tracks=[Track(track_number=1, title=None)],
        )
        err = self._run(monkeypatch, capsys, meta, tmp_path)
        assert (
            "Status:       needs review — missing album, album_artist, 1 track title"
            in err
        )

    def test_zero_tracks_renders_as_zero(self, tmp_path, monkeypatch, capsys):
        meta = make_meta(tracks=[])
        err = self._run(monkeypatch, capsys, meta, tmp_path)
        assert "Tracks:       0" in err
        # Zero tracks with album/album_artist populated is "complete" —
        # there are no track titles to be missing.
        assert "Status:       complete" in err


def test_main_full_flow(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "rips"
    audio.mkdir()
    out_path = tmp_path / "tags.json"
    meta = make_meta()
    monkeypatch.setattr(
        cdm, "load_local_tracks",
        lambda d: [{"position": 1, "filename": "x.flac", "duration_seconds": 1.0}],
    )
    monkeypatch.setattr(cdm, "fetch_page_text", lambda url: "PAGE")
    monkeypatch.setattr(cdm, "query_lmstudio", lambda *a, **k: meta)
    write_called: list = []
    rename_called: list = []
    monkeypatch.setattr(
        cdm, "write_tags",
        lambda d, m: write_called.append((d, m)),
    )
    # --rename returns a path → the default output should follow it,
    # but here --out is explicit so it should win.
    new_dir = tmp_path / "renamed"
    new_dir.mkdir()
    monkeypatch.setattr(
        cdm, "rename_release",
        lambda d, m: rename_called.append((d, m)) or new_dir,
    )
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py",
        "--url", "http://x",
        "--dir", str(audio),
        "--out", str(out_path),
        "--write",
        "--rename",
    ])
    main()
    captured = capsys.readouterr()
    assert captured.out == ""  # no more stdout JSON
    assert out_path.exists()
    assert json.loads(out_path.read_text())["album"] == "Some Album"
    assert write_called and rename_called
    assert f"Wrote:        {out_path}" in captured.err


def test_main_rename_redirects_default_out(tmp_path, monkeypatch, capsys):
    """Without --out, default landing follows the renamed directory."""
    audio = tmp_path / "rips"
    audio.mkdir()
    new_dir = tmp_path / "Some Artist - Some Album (2020) [FLAC]"
    new_dir.mkdir()
    meta = make_meta(date="2020")
    monkeypatch.setattr(cdm, "load_local_tracks", lambda d: [])
    monkeypatch.setattr(cdm, "fetch_page_text", lambda url: "PAGE")
    monkeypatch.setattr(cdm, "query_lmstudio", lambda *a, **k: meta)
    monkeypatch.setattr(cdm, "rename_release", lambda d, m: new_dir)
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--url", "http://x", "--dir", str(audio), "--rename",
    ])
    main()
    expected = new_dir / "metadata.json"
    assert expected.exists()
    assert f"Wrote:        {expected}" in capsys.readouterr().err


def test_main_rename_failure_falls_back_to_original_dir(tmp_path, monkeypatch):
    """If rename_release returns None, default --out stays in args.dir."""
    audio = tmp_path / "rips"
    audio.mkdir()
    meta = make_meta()
    monkeypatch.setattr(cdm, "load_local_tracks", lambda d: [])
    monkeypatch.setattr(cdm, "fetch_page_text", lambda url: "PAGE")
    monkeypatch.setattr(cdm, "query_lmstudio", lambda *a, **k: meta)
    monkeypatch.setattr(cdm, "rename_release", lambda d, m: None)
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--url", "http://x", "--dir", str(audio), "--rename",
    ])
    main()
    assert (audio / "metadata.json").exists()


# ---------- --metadata mode ----------

def _seed_metadata_json(audio_dir: Path, meta=None) -> Path:
    """Write a metadata.json into the audio dir so --metadata can reuse it."""
    meta = meta or make_meta(date="2020", label="Some Label")
    path = audio_dir / "metadata.json"
    path.write_text(meta.model_dump_json(indent=2))
    return path


def test_main_metadata_skips_lmstudio(tmp_path, monkeypatch, capsys, caplog):
    caplog.set_level(logging.INFO, logger="cd_metadata")
    audio = tmp_path / "rips"
    audio.mkdir()
    _seed_metadata_json(audio)

    # Make any accidental LM Studio path fail loudly.
    def boom(*a, **k):
        raise AssertionError("LM Studio path should not run in --metadata mode")
    monkeypatch.setattr(cdm, "load_local_tracks", boom)
    monkeypatch.setattr(cdm, "fetch_page_text", boom)
    monkeypatch.setattr(cdm, "query_lmstudio", boom)

    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--metadata", "--dir", str(audio),
    ])
    main()
    assert f"Loading metadata from {audio / 'metadata.json'}" in caplog.text
    err = capsys.readouterr().err
    assert "Album:        Some Album" in err
    assert "Year:         2020" in err


def test_main_metadata_applies_write_and_rename(tmp_path, monkeypatch):
    audio = tmp_path / "rips"
    audio.mkdir()
    _seed_metadata_json(audio)

    def boom(*a, **k):  # ensure LM Studio not consulted
        raise AssertionError("must not call LM Studio")
    monkeypatch.setattr(cdm, "fetch_page_text", boom)
    monkeypatch.setattr(cdm, "query_lmstudio", boom)
    write_called: list = []
    monkeypatch.setattr(
        cdm, "write_tags", lambda d, m: write_called.append((d, m))
    )
    new_dir = tmp_path / "Some Artist - Some Album (2020) [FLAC]"
    new_dir.mkdir()
    # Carry the seeded metadata file into the simulated renamed dir so the
    # post-rename write to <new_dir>/metadata.json reuses that path cleanly.
    (new_dir / "metadata.json").write_text(
        (audio / "metadata.json").read_text()
    )
    monkeypatch.setattr(cdm, "rename_release", lambda d, m: new_dir)

    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--metadata", "--dir", str(audio),
        "--write", "--rename",
    ])
    main()
    assert write_called and write_called[0][0] == audio
    assert (new_dir / "metadata.json").exists()


def test_main_metadata_missing_file_exits(tmp_path, monkeypatch):
    audio = tmp_path / "rips"
    audio.mkdir()  # no metadata.json inside
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--metadata", "--dir", str(audio),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert "No metadata file" in str(exc.value)


def test_main_metadata_cleans_disambiguators_on_load(tmp_path, monkeypatch, capsys):
    """If the on-disk JSON has wiki-style disambiguators, clean_metadata still strips them."""
    audio = tmp_path / "rips"
    audio.mkdir()
    dirty = make_meta(
        album="Discovery (album)",
        tracks=[Track(track_number=1, title="One More Time (song)")],
    )
    _seed_metadata_json(audio, meta=dirty)
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--metadata", "--dir", str(audio),
    ])
    main()
    on_disk = json.loads((audio / "metadata.json").read_text())
    assert on_disk["album"] == "Discovery"
    assert on_disk["tracks"][0]["title"] == "One More Time"


def test_main_requires_url_or_metadata(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "rips"
    audio.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--dir", str(audio),
    ])
    with pytest.raises(SystemExit):
        main()
    assert "one of the arguments" in capsys.readouterr().err.lower()


def test_main_url_and_metadata_mutually_exclusive(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "rips"
    audio.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--url", "http://x", "--metadata", "--dir", str(audio),
    ])
    with pytest.raises(SystemExit):
        main()
    assert "not allowed with" in capsys.readouterr().err.lower()


# ---------- --image mode ----------

def _fake_sips_run(cmd, **kwargs):
    """Fake subprocess.run for sips. Handles two call patterns:

    - dimensions query (`sips -g pixelWidth -g pixelHeight PATH`) → small dims
    - conversion (`sips ... --out PATH`) → write a stub JPEG to that path
    """
    if "-g" in cmd:
        return MagicMock(returncode=0, stdout=_sips_dims(800, 600), stderr="")
    out_path = Path(cmd[cmd.index("--out") + 1])
    out_path.write_bytes(b"FAKE JPEG")
    return MagicMock(returncode=0, stderr="", stdout="")


def test_main_image_skips_url_path(tmp_path, monkeypatch):
    audio = tmp_path / "rips"
    audio.mkdir()
    img = tmp_path / "back.jpg"
    img.write_bytes(b"")

    def boom(*a, **k):
        raise AssertionError("URL/text path should not run in --image mode")
    monkeypatch.setattr(cdm, "fetch_page_text", boom)
    monkeypatch.setattr(cdm, "query_lmstudio", boom)
    monkeypatch.setattr(cdm, "load_local_tracks", lambda d: [])
    monkeypatch.setattr(cdm.subprocess, "run", _fake_sips_run)
    captured: list = []
    monkeypatch.setattr(
        cdm, "query_lmstudio_image",
        lambda model, paths, tracks: captured.append((model, paths, tracks)) or make_meta(),
    )

    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--image", str(img), "--dir", str(audio),
    ])
    main()
    # query_lmstudio_image now receives a list of normalized JPEG temp paths.
    assert captured and len(captured[0][1]) == 1
    assert captured[0][1][0].suffix == ".jpg"
    assert (audio / "metadata.json").exists()


def test_main_image_multiple_inputs(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="cd_metadata")
    audio = tmp_path / "rips"
    audio.mkdir()
    a = tmp_path / "back.jpg"
    b = tmp_path / "insert.png"
    a.write_bytes(b"")
    b.write_bytes(b"")
    monkeypatch.setattr(cdm.subprocess, "run", _fake_sips_run)
    monkeypatch.setattr(cdm, "load_local_tracks", lambda d: [])
    seen: list = []
    monkeypatch.setattr(
        cdm, "query_lmstudio_image",
        lambda model, paths, tracks: seen.append(paths) or make_meta(),
    )
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--image", str(a), str(b), "--dir", str(audio),
    ])
    main()
    assert seen and len(seen[0]) == 2
    # Each input was normalized to its own JPEG temp file.
    assert {p.suffix for p in seen[0]} == {".jpg"}
    assert seen[0][0] != seen[0][1]
    # Per-image log lines.
    assert caplog.text.count("Normalized") == 2
    assert "Sending 2 image(s)" in caplog.text


def test_main_image_not_a_file_exits(tmp_path, monkeypatch):
    audio = tmp_path / "rips"
    audio.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py",
        "--image", str(tmp_path / "does-not-exist.jpg"),
        "--dir", str(audio),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert "Not a file" in str(exc.value)


def test_main_image_unsupported_format_exits(tmp_path, monkeypatch):
    audio = tmp_path / "rips"
    audio.mkdir()
    img = tmp_path / "back.tiff"
    img.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--image", str(img), "--dir", str(audio),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    msg = str(exc.value)
    assert "Unsupported image format" in msg
    assert ".tiff" in msg


def test_main_image_unsupported_format_among_multiple_exits(tmp_path, monkeypatch):
    audio = tmp_path / "rips"
    audio.mkdir()
    good = tmp_path / "back.jpg"
    bad = tmp_path / "insert.tiff"
    good.write_bytes(b"")
    bad.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py",
        "--image", str(good), str(bad),
        "--dir", str(audio),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert ".tiff" in str(exc.value)


# ---------- _resolve_image (sips normalization) ----------


def _sips_dims(width: int, height: int):
    """Build a fake sips -g output string for the given dimensions.

    Mirrors the real sips output: an unindented path line followed by
    indented `key: value` lines. Covering the path-line branch matters
    because cd_metadata's parser must skip non-matching keys.
    """
    return f"/tmp/fake.jpg\n  pixelWidth: {width}\n  pixelHeight: {height}\n"


class TestResolveImage:
    def test_normalizes_small_jpeg_without_resize(self, tmp_path, monkeypatch):
        """Small images skip the -Z flag (sips -Z would otherwise upscale)."""
        jpg = tmp_path / "back.jpg"
        jpg.write_bytes(b"data")
        captured: list = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            if "-g" in cmd:
                return MagicMock(
                    returncode=0, stdout=_sips_dims(800, 600), stderr="",
                )
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_bytes(b"NORMALIZED")
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(cdm.subprocess, "run", fake_run)
        with cdm._resolve_image(jpg) as resolved:
            assert resolved.suffix == ".jpg"
            assert resolved != jpg  # temp file, not the original
            assert resolved.read_bytes() == b"NORMALIZED"
        assert not resolved.exists()
        # Conversion call carries the format flag but NOT -Z (image fits).
        convert_cmd = captured[-1]
        assert convert_cmd[:4] == ["sips", "-s", "format", "jpeg"]
        assert "-Z" not in convert_cmd
        assert str(jpg) in convert_cmd

    def test_resizes_oversized_jpeg(self, tmp_path, monkeypatch):
        """Images exceeding IMAGE_MAX_DIMENSION get the -Z cap."""
        jpg = tmp_path / "big.jpg"
        jpg.write_bytes(b"")
        captured: list = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            if "-g" in cmd:
                return MagicMock(
                    returncode=0, stdout=_sips_dims(4032, 3024), stderr="",
                )
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_bytes(b"RESIZED")
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(cdm.subprocess, "run", fake_run)
        with cdm._resolve_image(jpg) as resolved:
            assert resolved.read_bytes() == b"RESIZED"
        convert_cmd = captured[-1]
        assert "-Z" in convert_cmd
        assert convert_cmd[convert_cmd.index("-Z") + 1] == str(cdm.IMAGE_MAX_DIMENSION)

    def test_converts_heic_via_sips(self, tmp_path, monkeypatch):
        heic = tmp_path / "back.heic"
        heic.write_bytes(b"")
        captured: list = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            if "-g" in cmd:
                return MagicMock(
                    returncode=0, stdout=_sips_dims(1024, 768), stderr="",
                )
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_bytes(b"FAKE JPEG")
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(cdm.subprocess, "run", fake_run)
        with cdm._resolve_image(heic) as resolved:
            assert resolved.suffix == ".jpg"
            assert resolved.read_bytes() == b"FAKE JPEG"
            assert resolved.exists()
        # Cleanup after exit
        assert not resolved.exists()
        # Format conversion happens on the second sips call.
        convert_cmd = captured[-1]
        assert convert_cmd[:4] == ["sips", "-s", "format", "jpeg"]
        assert str(heic) in convert_cmd

    def test_converts_heif_too(self, tmp_path, monkeypatch):
        heif = tmp_path / "back.heif"
        heif.write_bytes(b"")
        monkeypatch.setattr(cdm.subprocess, "run", _fake_sips_run)
        with cdm._resolve_image(heif) as resolved:
            assert resolved.suffix == ".jpg"

    def test_cleans_up_on_exception(self, tmp_path, monkeypatch):
        heic = tmp_path / "back.heic"
        heic.write_bytes(b"")
        monkeypatch.setattr(cdm.subprocess, "run", _fake_sips_run)
        captured: list = []
        with pytest.raises(RuntimeError):
            with cdm._resolve_image(heic) as resolved:
                captured.append(resolved)
                raise RuntimeError("downstream blew up")
        assert captured and not captured[0].exists()

    def test_sips_not_found_exits(self, tmp_path, monkeypatch):
        heic = tmp_path / "back.heic"
        heic.write_bytes(b"")

        def fake_run(*a, **k):
            raise FileNotFoundError("no sips")

        monkeypatch.setattr(cdm.subprocess, "run", fake_run)
        with pytest.raises(SystemExit) as exc:
            with cdm._resolve_image(heic):
                pass
        assert "sips not found" in str(exc.value)

    def test_sips_failure_exits(self, tmp_path, monkeypatch):
        heic = tmp_path / "back.heic"
        heic.write_bytes(b"")

        def fake_run(cmd, **kwargs):
            if "-g" in cmd:
                return MagicMock(
                    returncode=0, stdout=_sips_dims(800, 600), stderr="",
                )
            return MagicMock(returncode=1, stderr="bad input", stdout="")

        monkeypatch.setattr(cdm.subprocess, "run", fake_run)
        with pytest.raises(SystemExit) as exc:
            with cdm._resolve_image(heic):
                pass
        assert "sips failed" in str(exc.value)
        assert "bad input" in str(exc.value)

    def test_sips_failure_with_empty_output(self, tmp_path, monkeypatch):
        """Cover the '(no message)' fallback when sips emits no stderr/stdout."""
        heic = tmp_path / "back.heic"
        heic.write_bytes(b"")

        def fake_run(cmd, **kwargs):
            if "-g" in cmd:
                return MagicMock(
                    returncode=0, stdout=_sips_dims(800, 600), stderr="",
                )
            return MagicMock(returncode=1, stderr="", stdout="")

        monkeypatch.setattr(cdm.subprocess, "run", fake_run)
        with pytest.raises(SystemExit) as exc:
            with cdm._resolve_image(heic):
                pass
        assert "(no message)" in str(exc.value)

    def test_dim_query_failure_still_resizes(self, tmp_path, monkeypatch):
        """If sips -g fails, fall back to including -Z so the convert call
        surfaces the real error (don't silently skip the cap)."""
        jpg = tmp_path / "back.jpg"
        jpg.write_bytes(b"")
        captured: list = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            if "-g" in cmd:
                return MagicMock(returncode=1, stdout="", stderr="bad header")
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_bytes(b"X")
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(cdm.subprocess, "run", fake_run)
        with cdm._resolve_image(jpg):
            pass
        # -Z is included because the dim query failed.
        assert "-Z" in captured[-1]

    def test_dim_query_non_numeric_treated_as_oversized(self, tmp_path, monkeypatch):
        """Garbage pixel-dimension output → safer to include -Z than skip it."""
        jpg = tmp_path / "back.jpg"
        jpg.write_bytes(b"")
        captured: list = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            if "-g" in cmd:
                return MagicMock(
                    returncode=0,
                    stdout="  pixelWidth: not-a-number\n  pixelHeight: 600\n",
                    stderr="",
                )
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_bytes(b"X")
            return MagicMock(returncode=0, stderr="", stdout="")

        monkeypatch.setattr(cdm.subprocess, "run", fake_run)
        with cdm._resolve_image(jpg):
            pass
        assert "-Z" in captured[-1]


def test_main_image_heic_runs_sips_and_passes_jpg(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="cd_metadata")
    audio = tmp_path / "rips"
    audio.mkdir()
    heic = tmp_path / "back.heic"
    heic.write_bytes(b"")

    monkeypatch.setattr(cdm.subprocess, "run", _fake_sips_run)
    monkeypatch.setattr(cdm, "load_local_tracks", lambda d: [])
    seen: list = []
    monkeypatch.setattr(
        cdm, "query_lmstudio_image",
        lambda model, paths, tracks: seen.append(paths) or make_meta(),
    )

    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py", "--image", str(heic), "--dir", str(audio),
    ])
    main()
    assert seen and len(seen[0]) == 1
    assert seen[0][0].suffix == ".jpg"
    assert seen[0][0] != heic  # normalized temp path, not the original
    assert "Normalized" in caplog.text and "sips" in caplog.text


def test_main_url_and_image_mutually_exclusive(tmp_path, monkeypatch, capsys):
    audio = tmp_path / "rips"
    audio.mkdir()
    img = tmp_path / "back.jpg"
    img.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", [
        "cd_metadata.py",
        "--url", "http://x",
        "--image", str(img),
        "--dir", str(audio),
    ])
    with pytest.raises(SystemExit):
        main()
    assert "not allowed with" in capsys.readouterr().err.lower()
