import random

import pytest

from discord_bot.persistence.models import Participant
from discord_bot.rooms.turn_policies import (
    DebatePolicy,
    ManualPolicy,
    ModeratedPolicy,
    ReplyPolicy,
    RoundRobinPolicy,
    Trigger,
    TurnPolicyError,
    build_policy,
)


def mkp(participant_id, actor_id=None, suffix="a"):
    return Participant(
        id=participant_id, room_id="room-1", actor_id=actor_id or participant_id,
        instance_suffix=suffix, joined_at="t", active=True,
    )


PARTICIPANTS = [mkp("weil"), mkp("hugo"), mkp("dostoevsky")]


def decide(policy, participants=PARTICIPANTS, cursor=None, last_speaker_id=None,
           trigger=Trigger(), rng=None):
    return policy.decide(
        participants=participants, cursor=cursor or {}, last_speaker_id=last_speaker_id,
        trigger=trigger, rng=rng or random.Random(0),
    )


# -- manual -----------------------------------------------------------------

def test_manual_requires_explicit_participant():
    with pytest.raises(TurnPolicyError, match="requires an explicit participant"):
        decide(ManualPolicy())


def test_manual_returns_the_requested_participant():
    d = decide(ManualPolicy(), trigger=Trigger(requested_participant_id="hugo"))
    assert d.participant_id == "hugo"
    assert not d.done


def test_manual_rejects_an_inactive_or_unknown_participant():
    with pytest.raises(TurnPolicyError, match="not an active participant"):
        decide(ManualPolicy(), trigger=Trigger(requested_participant_id="nobody"))


# -- reply --------------------------------------------------------------------

def test_reply_requires_an_addressed_participant():
    with pytest.raises(TurnPolicyError, match="requires an addressed participant"):
        decide(ReplyPolicy())


def test_reply_returns_the_addressed_participant():
    d = decide(ReplyPolicy(), trigger=Trigger(requested_participant_id="weil"))
    assert d.participant_id == "weil"


# -- round robin --------------------------------------------------------------

def test_round_robin_cycles_in_join_order_and_wraps():
    policy = RoundRobinPolicy()
    cursor = {}
    seen = []
    for _ in range(6):  # two full laps over 3 participants
        d = decide(policy, cursor=cursor)
        seen.append(d.participant_id)
        cursor = d.next_cursor
    assert seen == ["weil", "hugo", "dostoevsky", "weil", "hugo", "dostoevsky"]


def test_round_robin_requires_active_participants():
    with pytest.raises(TurnPolicyError, match="no active participants"):
        decide(RoundRobinPolicy(), participants=[])


# -- moderated ------------------------------------------------------------------

def test_moderated_never_repeats_the_last_speaker_when_others_are_available():
    policy = ModeratedPolicy()
    rng = random.Random(1)
    for _ in range(20):
        d = decide(policy, last_speaker_id="weil", rng=rng)
        assert d.participant_id != "weil"


def test_moderated_falls_back_to_the_only_participant_even_if_they_spoke_last():
    policy = ModeratedPolicy()
    d = decide(policy, participants=[mkp("weil")], last_speaker_id="weil")
    assert d.participant_id == "weil"


def test_moderated_requires_active_participants():
    with pytest.raises(TurnPolicyError, match="no active participants"):
        decide(ModeratedPolicy(), participants=[])


# -- debate -----------------------------------------------------------------

def test_debate_requires_order_and_rounds():
    with pytest.raises(TurnPolicyError, match="'order'"):
        DebatePolicy({"rounds": 2})
    with pytest.raises(TurnPolicyError, match="'rounds'"):
        DebatePolicy({"order": ["weil", "hugo"]})
    with pytest.raises(TurnPolicyError, match="'rounds'"):
        DebatePolicy({"order": ["weil"], "rounds": 0})


def test_debate_cycles_fixed_order_for_exactly_rounds_then_is_done():
    policy = DebatePolicy({"order": ["weil", "hugo"], "rounds": 2})
    cursor = {}
    seen = []
    for _ in range(4):
        d = decide(policy, cursor=cursor)
        seen.append(d.participant_id)
        assert not d.done
        cursor = d.next_cursor
    assert seen == ["weil", "hugo", "weil", "hugo"]

    finished = decide(policy, cursor=cursor)
    assert finished.done
    assert finished.participant_id is None


def test_debate_rejects_a_participant_missing_from_the_room():
    policy = DebatePolicy({"order": ["weil", "someone-else"], "rounds": 1})
    with pytest.raises(TurnPolicyError, match="not an active participant"):
        decide(policy, participants=[mkp("weil")])


# -- build_policy -------------------------------------------------------------

def test_build_policy_known_names():
    assert build_policy("manual").name == "manual"
    assert build_policy("reply").name == "reply"
    assert build_policy("round_robin").name == "round_robin"
    assert build_policy("moderated").name == "moderated"
    debate = build_policy("debate", {"order": ["weil"], "rounds": 1})
    assert debate.name == "debate"


def test_build_policy_unknown_name_raises():
    with pytest.raises(TurnPolicyError, match="unknown turn policy"):
        build_policy("made-up")
