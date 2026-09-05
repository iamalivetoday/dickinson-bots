import pytest

from discord_bot.discord_app.config import ConfigError, load_settings


def test_missing_token_raises():
    with pytest.raises(ConfigError, match="DISCORD_BOT_TOKEN"):
        load_settings({}, env_file=None)


def test_minimal_settings_have_sane_defaults():
    settings = load_settings({"DISCORD_BOT_TOKEN": "tok"}, env_file=None)
    assert settings.discord_bot_token == "tok"
    assert settings.anthropic_api_key is None
    assert settings.allowed_guild_id is None
    assert settings.allowed_user_ids == frozenset()
    assert settings.local_max_cached_bases == 1
    assert settings.local_device is None
    assert str(settings.actors_config_path).endswith("config/actors.yaml")


def test_allowed_user_ids_parses_a_comma_separated_list():
    settings = load_settings(
        {"DISCORD_BOT_TOKEN": "tok", "DISCORD_ALLOWED_USER_IDS": "111, 222,333"},
        env_file=None,
    )
    assert settings.allowed_user_ids == frozenset({"111", "222", "333"})


def test_blank_allowed_user_ids_means_unrestricted():
    settings = load_settings(
        {"DISCORD_BOT_TOKEN": "tok", "DISCORD_ALLOWED_USER_IDS": ""}, env_file=None
    )
    assert settings.allowed_user_ids == frozenset()


def test_explicit_values_override_defaults():
    settings = load_settings(
        {
            "DISCORD_BOT_TOKEN": "tok",
            "DISCORD_GUILD_ID": "g1",
            "ANTHROPIC_API_KEY": "sk-ant-x",
            "LOCAL_MAX_CACHED_BASES": "3",
            "LOCAL_DEVICE": "cpu",
            "ROOM_DB_PATH": "/tmp/rooms.sqlite3",
            "ACTORS_CONFIG_PATH": "/tmp/actors.yaml",
        },
        env_file=None,
    )
    assert settings.allowed_guild_id == "g1"
    assert settings.anthropic_api_key == "sk-ant-x"
    assert settings.local_max_cached_bases == 3
    assert settings.local_device == "cpu"
    assert str(settings.room_db_path) == "/tmp/rooms.sqlite3"
    assert str(settings.actors_config_path) == "/tmp/actors.yaml"


def test_missing_env_file_is_not_an_error(tmp_path):
    settings = load_settings(
        {"DISCORD_BOT_TOKEN": "tok"}, env_file=tmp_path / "does-not-exist.env"
    )
    assert settings.discord_bot_token == "tok"
