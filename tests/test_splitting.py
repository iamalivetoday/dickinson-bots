import pytest

from discord_bot.discord_app.splitting import split_message


def test_empty_text_splits_to_nothing():
    assert split_message("") == []
    assert split_message("   ") == []


def test_text_under_the_limit_is_returned_unchanged():
    assert split_message("hello world", limit=2000) == ["hello world"]


def test_no_chunk_ever_exceeds_the_limit():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = split_message(text, limit=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_splits_prefer_whitespace_over_mid_word():
    text = "aaaa bbbb cccc dddd eeee"
    chunks = split_message(text, limit=12)
    assert all(not c.startswith(" ") and not c.endswith(" ") for c in chunks)
    # every emitted word survives whole and in order — proves no mid-word split
    assert " ".join(chunks).split() == text.split()


def test_splits_prefer_paragraph_breaks_when_available():
    text = "First paragraph here.\n\n" + ("x" * 50) + "\n\nThird paragraph."
    chunks = split_message(text, limit=40)
    assert chunks[0] == "First paragraph here."


def test_a_single_token_longer_than_the_limit_is_hard_split():
    text = "a" * 250
    chunks = split_message(text, limit=100)
    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_rejects_a_non_positive_limit():
    with pytest.raises(ValueError):
        split_message("hi", limit=0)


def test_default_limit_matches_discords_message_cap():
    text = "x" * 2500
    chunks = split_message(text)
    assert len(chunks) == 2
    assert all(len(c) <= 2000 for c in chunks)
