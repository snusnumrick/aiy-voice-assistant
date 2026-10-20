import json
import logging

import pytest

from src.response_metadata import (
    iter_service_tags,
    split_complete_meta_tag_prefix,
    strip_service_metadata,
)

EMOTION = '{"light":{"color":[255,160,80],"behavior":"breathing","brightness":"medium","period":2},"voice":{"tone":"plain"}}'
SMART_EMOTION = EMOTION.replace('"light"', '“light”').replace('"plain"', '“plain”')
OLD_PROMPT_EMOTION = EMOTION.replace('},"voice"', '}}, "voice"')
SMART_OLD_PROMPT_EMOTION = SMART_EMOTION.replace('},"voice"', '}}, "voice"')


@pytest.mark.parametrize("payload", [EMOTION, SMART_EMOTION, OLD_PROMPT_EMOTION, SMART_OLD_PROMPT_EMOTION, '{"light":invalid}'])
@pytest.mark.parametrize("delimiter", ["$", ""])
def test_balanced_metadata_is_removed_without_requiring_valid_json(payload, delimiter):
    text = f"Before. $emotion:{payload}{delimiter} Neutral reply."
    assert strip_service_metadata(text) == "Before.  Neutral reply."


@pytest.mark.parametrize("payload", [
    '{"light":',
    '{"light":{"color":[255,',
    '{"voice":{"tone":"unfinished $5 and $lang:en$ text',
    '{“voice”:{“tone”:“unfinished $lang:en$ text',
    '{"light": invalid $5 and hidden payload',
    "invalid payload without a terminator",
    "",
])
def test_ambiguous_emotion_tail_is_discarded(payload):
    assert strip_service_metadata("Before. $emotion:" + payload) == "Before. "


@pytest.mark.parametrize("text,expected", [
    ('$emotion:{"light":$ Neutral reply.', ' Neutral reply.'),
    ('$emotion:invalid$ Neutral reply.', ' Neutral reply.'),
    ('$emotion:$ Neutral reply.', ' Neutral reply.'),
    ('$emotion:{}$lang: en$ Neutral reply.', ' Neutral reply.'),
    ('$emotion:{} $5.50 and $10 today.', ' $5.50 and $10 today.'),
    ('$lang: en$ It costs $5.50 and $10.', ' It costs $5.50 and $10.'),
    ('Pay $5, $10 or US$20; use $HOME. A literal $.', 'Pay $5, $10 or US$20; use $HOME. A literal $.'),
    ('$speaker: Tester$ $remember: neutral fact$ $rule: neutral rule$ Reply.', '   Reply.'),
    ('$tool_filler: Please wait.$', 'Please wait.'),
    ('$rule: use $emotion:...$ when useful.$ Reply.', ' Reply.'),
    ('Before. $emo', 'Before. '),
    ('Before. $tool_fil', 'Before. '),
])
def test_known_tags_and_currency(text, expected):
    assert strip_service_metadata(text) == expected


@pytest.mark.parametrize("payload", [EMOTION, SMART_EMOTION, OLD_PROMPT_EMOTION, SMART_OLD_PROMPT_EMOTION, '{"light":invalid}'])
@pytest.mark.parametrize("delimiter", ["$", ""])
def test_every_two_chunk_split_and_single_character_stream(payload, delimiter):
    text = f"$lang: en$ Before. $emotion:{payload}{delimiter} Neutral reply costs $5.50."
    chunkings = [[text[:cut], text[cut:]] for cut in range(len(text) + 1)] + [list(text)]
    expected = strip_service_metadata(text)
    for chunks in chunkings:
        pending = ""
        ready = []
        for chunk in chunks:
            prefix, pending = split_complete_meta_tag_prefix(pending + chunk)
            ready.append(prefix)
        prefix, pending = split_complete_meta_tag_prefix(pending, final=True)
        ready.append(prefix)
        assert not pending
        assert strip_service_metadata("".join(ready)) == expected, chunks


def test_dollars_braces_and_escaped_quotes_inside_json_strings_do_not_end_metadata():
    payload = json.dumps({"voice": {"tone": 'a $5 quote " with } and $emotion:{}$'}})
    text = f"$emotion:{payload}$ Neutral reply."
    tags = list(iter_service_tags(text))
    assert len(tags) == 1
    assert json.loads(tags[0].payload)["voice"]["tone"].startswith("a $5 quote")
    assert strip_service_metadata(text) == " Neutral reply."


def test_currency_is_emitted_before_end_of_stream():
    ready, pending = split_complete_meta_tag_prefix("It costs $5 today. ")
    assert ready == "It costs $5 today. "
    assert pending == ""


def test_a_word_containing_punctuation_is_not_split_inside_a_chunk():
    ready, pending = split_complete_meta_tag_prefix("Visit example.com")
    assert ready == "Visit "
    assert pending == "example.com"


@pytest.mark.parametrize("text,tag_type,reasons", [
    ('$emotion:{} Neutral reply.', 'emotion', ['missing_delimiter']),
    ('$emotion:{"voice":invalid}$ Neutral reply.', 'emotion', ['invalid_json']),
    ('$emotion:{“voice”:{“tone”:“plain”}} Neutral reply.', 'emotion', ['missing_delimiter', 'invalid_json', 'typographic_quotes']),
    ('$emotion:{"voice":', 'emotion', ['truncated_metadata']),
    ('$emo', 'emotion', ['truncated_metadata']),
    ('$speaker: unfinished', 'speaker', ['truncated_metadata']),
    ('$remember: unfinished', 'remember', ['truncated_metadata']),
    ('$rule: unfinished', 'rule', ['truncated_metadata']),
    ('$tool_filler: unfinished', 'tool_filler', ['truncated_metadata']),
    ('$lang: en Neutral reply.', 'lang', ['missing_delimiter']),
    ('$lang:123$ Neutral reply.', 'lang', ['invalid_language']),
    ('$emotion:[]$ Neutral reply.', 'emotion', ['invalid_payload_type']),
])
def test_removed_malformed_metadata_logs_one_private_safe_error(caplog, text, tag_type, reasons):
    with caplog.at_level(logging.ERROR, logger="src.response_metadata"):
        strip_service_metadata("CONFIDENTIAL_PROSE " + text)
    records = [record for record in caplog.records if record.name == "src.response_metadata"]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].getMessage() == (
        f"Malformed service metadata removed: tag={tag_type} reason={','.join(reasons)}"
    )
    assert "CONFIDENTIAL_PROSE" not in caplog.text
    assert "Neutral reply" not in caplog.text
    assert "tone" not in caplog.text


def test_valid_tags_and_currency_produce_no_metadata_errors(caplog):
    text = (
        '$lang: en$ $emotion:{"voice":{"tone":"plain","description":"“quoted” $5"}}$ '
        '$speaker: Tester$ $remember: neutral fact$ $rule: neutral rule$ '
        '$tool_filler: Please wait.$ Reply costs $5 and $10.'
    )
    with caplog.at_level(logging.ERROR, logger="src.response_metadata"):
        strip_service_metadata(text)
        pending = ""
        for char in text:
            _, pending = split_complete_meta_tag_prefix(pending + char)
        split_complete_meta_tag_prefix(pending, final=True)
    assert not [record for record in caplog.records if record.name == "src.response_metadata"]


def test_speculative_scans_are_silent_and_truncation_is_logged_only_at_eof(caplog):
    text = '$emotion:{"voice":{"tone":"CONFIDENTIAL_PAYLOAD'
    with caplog.at_level(logging.ERROR, logger="src.response_metadata"):
        list(iter_service_tags(text))
        pending = ""
        for char in text:
            ready, pending = split_complete_meta_tag_prefix(pending + char)
            assert ready == ""
        assert not caplog.records
        ready, pending = split_complete_meta_tag_prefix(pending, final=True)
    assert (ready, pending) == ("", "")
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage().endswith("tag=emotion reason=truncated_metadata")
    assert "CONFIDENTIAL_PAYLOAD" not in caplog.text


def test_two_identical_malformed_tags_are_two_events(caplog):
    with caplog.at_level(logging.ERROR, logger="src.response_metadata"):
        strip_service_metadata('$emotion:{bad}$ One. $emotion:{bad}$ Two.')
    assert len(caplog.records) == 2
