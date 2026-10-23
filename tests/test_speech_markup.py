import logging

import pytest

from src.response_metadata import split_complete_meta_tag_prefix
from src.speech_markup import (
    PAUSE_SIZES,
    normalize_pause_markup,
    pause_instructions,
    pause_safe_split,
    render_pause_markup,
    strip_pause_markup,
)


@pytest.mark.parametrize("size,seconds", zip(PAUSE_SIZES, ("0.1", "0.3", "0.6", "1.0", "2.0")))
def test_provider_rendering_preserves_yandex_and_translates_elevenlabs(size, seconds):
    text = f"Before. <[{size}]> After."
    assert render_pause_markup(text, "yandex") == text
    assert render_pause_markup(text, "ssml") == f'Before. <break time="{seconds}s" /> After.'
    assert "<[" not in render_pause_markup(text, "audio_tags")
    assert render_pause_markup(text, "plain") == "Before.   After."
    assert strip_pause_markup(text) == "Before.  After."
    for style in ("yandex", "ssml", "audio_tags", "plain"):
        rendered = render_pause_markup(text, style)
        assert render_pause_markup(rendered, style) == rendered


@pytest.mark.parametrize("text,reason", [
    ("Before. <[small", "truncated_metadata"),
    ("Before. <[unknown]> After.", "unknown_pause_size"),
    ("Before. <[small> After.", "missing_bracket"),
    ("Before. <small>[small]</small> After.", "html_pause_wrapper"),
])
def test_malformed_cues_are_hidden_or_recovered_with_one_private_safe_error(caplog, text, reason):
    with caplog.at_level(logging.ERROR, logger="src.speech_markup"):
        normalized = normalize_pause_markup(text)
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == f"Malformed service metadata removed: tag=pause reason={reason}"
    assert "Before" not in caplog.text and "After" not in caplog.text
    assert "unknown" not in normalized
    assert "<small>" not in normalized


def test_pause_only_requests_are_not_sent_and_plain_speech_is_unchanged():
    for style in ("yandex", "ssml", "audio_tags", "plain"):
        assert render_pause_markup(" <[small]> <[large]> ", style) == ""
        assert render_pause_markup("It costs $5; 2 < 3. Keep small words.", style) == "It costs $5; 2 < 3. Keep small words."


@pytest.mark.parametrize("tag", ["<[small]>", "<[ small ]>", "<small>[small]</small>"])
def test_pause_markup_survives_every_stream_boundary(tag):
    text = f"$lang: ru$ До. {tag} После. $lang: en$ Before. <[small]> After."
    chunkings = [[text[:cut], text[cut:]] for cut in range(len(text) + 1)] + [list(text)]
    for chunks in chunkings:
        pending = ""
        ready = []
        for chunk in chunks:
            prefix, pending = split_complete_meta_tag_prefix(pending + chunk, log_errors=False)
            ready.append(prefix)
        prefix, _ = split_complete_meta_tag_prefix(pending, final=True, log_errors=False)
        ready.append(prefix)
        assert "".join(ready) == text


def test_truncated_pause_is_discarded_only_at_eof(caplog):
    pending = ""
    ready = []
    with caplog.at_level(logging.ERROR, logger="src.speech_markup"):
        for char in "Before. <[ sma":
            prefix, pending = split_complete_meta_tag_prefix(pending + char)
            ready.append(prefix)
        assert not caplog.records
        prefix, _ = split_complete_meta_tag_prefix(pending, final=True)
        ready.append(prefix)
    assert "".join(ready) == "Before. "
    assert len(caplog.records) == 1


def test_pause_strings_inside_hidden_json_are_not_treated_as_truncated_speech():
    text = '$emotion:{"voice":{"description":"<[sm"}}$ Spoken.'
    assert split_complete_meta_tag_prefix(text, final=True, log_errors=False) == (text, "")


def test_rendered_pause_tokens_cannot_be_cut_at_internal_spaces():
    tag = '<break time="0.3s" />'
    text = "word " + tag + " afterwards"
    for at in range(6, 5 + len(tag)):
        assert pause_safe_split(text, at, len(text)) == 5
    assert pause_safe_split(tag + "words", 6, len(tag)) == len(tag)
    with pytest.raises(ValueError):
        pause_safe_split(tag, 6, 8)


def test_prompt_uses_one_application_format_for_all_spoken_languages():
    for language in ("russian", "english"):
        rules = pause_instructions(language)
        assert all(f"<[{size}]>" in rules for size in PAUSE_SIZES)
        assert "<break" not in rules and "[short pause]" not in rules
    assert "any language" in pause_instructions("english")


def test_native_yandex_emphasis_is_not_misclassified_as_a_malformed_pause(caplog):
    text = "An <[accented]> important word."
    assert render_pause_markup(text, "yandex") == text
    assert "<[accented]>" not in render_pause_markup(text, "ssml")
    assert not caplog.records
