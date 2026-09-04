import pytest

from discord_bot.registry import ActorConfigError, ActorRegistry


def write_yaml(path, text):
    path.write_text(text)
    return path


def test_loads_real_actors_yaml():
    registry = ActorRegistry.load()
    assert "weil" in registry
    assert "opus-4.8" in registry
    weil = registry.get("weil")
    assert weil.backend == "local"
    assert weil.persona == "weil"
    assert weil.base == "Qwen/Qwen2.5-7B-Instruct"
    assert weil.adapter_path().name == "chat_model"

    opus = registry.get("opus-4.8")
    assert opus.backend == "anthropic"
    assert opus.persona is None
    assert opus.model == "claude-opus-4-8"


def test_generation_defaults_merge_with_per_actor_overrides(tmp_path):
    cfg = write_yaml(tmp_path / "actors.yaml", """
defaults:
  local:
    temperature: 0.7
    max_new_tokens: 100
actors:
  - id: a
    backend: local
    base: some/base
    generation:
      temperature: 0.9
""")
    registry = ActorRegistry.load(cfg)
    a = registry.get("a")
    assert a.generation == {"temperature": 0.9, "max_new_tokens": 100}


def test_persona_less_base_model_actor_is_allowed(tmp_path):
    cfg = write_yaml(tmp_path / "actors.yaml", """
actors:
  - id: base-only
    backend: local
    base: some/base
""")
    registry = ActorRegistry.load(cfg)
    assert registry.get("base-only").persona is None


def test_duplicate_actor_id_rejected(tmp_path):
    cfg = write_yaml(tmp_path / "actors.yaml", """
actors:
  - id: dup
    backend: fake
  - id: dup
    backend: fake
""")
    with pytest.raises(ActorConfigError, match="duplicate actor id"):
        ActorRegistry.load(cfg)


def test_backend_is_required(tmp_path):
    cfg = write_yaml(tmp_path / "actors.yaml", """
actors:
  - id: no-backend
""")
    with pytest.raises(ActorConfigError, match="missing a string 'backend'"):
        ActorRegistry.load(cfg)


def test_registry_is_substrate_agnostic_any_backend_name_is_accepted(tmp_path):
    """The registry doesn't hardcode which backends exist — a not-yet-wired
    backend name is a router/main-wiring concern, not a config error, so
    onboarding a new backend never means teaching the registry about it."""
    cfg = write_yaml(tmp_path / "actors.yaml", """
actors:
  - id: x
    backend: some-future-backend
""")
    registry = ActorRegistry.load(cfg)
    assert registry.get("x").backend == "some-future-backend"


def test_empty_registry_rejected(tmp_path):
    cfg = write_yaml(tmp_path / "actors.yaml", "actors: []\n")
    with pytest.raises(ActorConfigError, match="no actors declared"):
        ActorRegistry.load(cfg)


def test_unknown_field_rejected(tmp_path):
    cfg = write_yaml(tmp_path / "actors.yaml", """
actors:
  - id: x
    backend: fake
    nonsense_field: 1
""")
    with pytest.raises(ActorConfigError):
        ActorRegistry.load(cfg)


def test_registry_iteration_and_ids(tmp_path):
    cfg = write_yaml(tmp_path / "actors.yaml", """
actors:
  - id: b
    backend: fake
  - id: a
    backend: fake
""")
    registry = ActorRegistry.load(cfg)
    assert registry.ids() == ["a", "b"]
    assert len(registry) == 2
    assert {a.id for a in registry} == {"a", "b"}
    with pytest.raises(KeyError):
        registry.get("missing")
