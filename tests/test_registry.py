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


def test_local_backend_requires_base(tmp_path):
    cfg = write_yaml(tmp_path / "actors.yaml", """
actors:
  - id: no-base
    backend: local
""")
    with pytest.raises(ActorConfigError, match="requires 'base'"):
        ActorRegistry.load(cfg)


def test_anthropic_backend_requires_model(tmp_path):
    cfg = write_yaml(tmp_path / "actors.yaml", """
actors:
  - id: no-model
    backend: anthropic
""")
    with pytest.raises(ActorConfigError, match="requires 'model'"):
        ActorRegistry.load(cfg)


def test_unknown_backend_rejected(tmp_path):
    cfg = write_yaml(tmp_path / "actors.yaml", """
actors:
  - id: x
    backend: made-up
""")
    with pytest.raises(ActorConfigError, match="backend must be one of"):
        ActorRegistry.load(cfg)


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
