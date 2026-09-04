from discord_bot.backends.base import TranscriptMessage, system_prompt_for, to_role_messages
from discord_bot.registry import Actor


def msg(speaker_id, speaker_name, content):
    return TranscriptMessage(speaker_id=speaker_id, speaker_name=speaker_name, content=content)


def test_own_lines_are_assistant_others_are_user_tagged_with_name():
    transcript = [
        msg("user:1", "Madeleine", "What is attention?"),
        msg("weil", "Simone Weil", "Attention is the rarest form of generosity."),
        msg("hugo", "Victor Hugo", "And love its highest expression!"),
    ]
    rendered = to_role_messages("weil", transcript)
    assert rendered == [
        {"role": "user", "content": "Madeleine: What is attention?"},
        {"role": "assistant", "content": "Attention is the rarest form of generosity."},
        {"role": "user", "content": "Victor Hugo: And love its highest expression!"},
    ]


def test_consecutive_same_role_turns_collapse():
    transcript = [
        msg("hugo", "Victor Hugo", "First."),
        msg("dostoevsky", "Fyodor Dostoevsky", "Second."),
        msg("weil", "Simone Weil", "Third, my own reply."),
    ]
    rendered = to_role_messages("weil", transcript)
    assert rendered == [
        {"role": "user", "content": "Victor Hugo: First.\n\nFyodor Dostoevsky: Second."},
        {"role": "assistant", "content": "Third, my own reply."},
    ]


def test_empty_transcript_seeds_a_user_turn():
    assert to_role_messages("weil", []) == [
        {"role": "user", "content": "(the conversation begins)"}
    ]


def test_transcript_starting_with_own_voice_gets_a_seed_user_turn_prepended():
    transcript = [msg("weil", "Simone Weil", "I spoke first, unaddressed.")]
    rendered = to_role_messages("weil", transcript)
    assert rendered[0] == {"role": "user", "content": "(the conversation begins)"}
    assert rendered[1] == {"role": "assistant", "content": "I spoke first, unaddressed."}


def test_system_prompt_uses_persona_voice():
    actor = Actor(id="weil", backend="local", persona="weil", base="some/base")
    assert "Simone Weil" in system_prompt_for(actor)


def test_system_prompt_falls_back_for_persona_less_actor():
    actor = Actor(id="opus-4.8", backend="anthropic", model="claude-opus-4-8")
    prompt = system_prompt_for(actor)
    assert "opus-4.8" in prompt
