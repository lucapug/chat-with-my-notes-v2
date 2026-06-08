from pathlib import Path

from config import Settings, settings


def test_settings_loads_dotenv_from_config_dir() -> None:
    env_file = Settings.model_config["env_file"]
    env_path = Path(env_file)

    assert env_path.is_absolute()
    assert env_path.name == ".env"
    assert env_path.parent == Path(__file__).resolve().parent.parent


def test_settings_uses_updated_generation_model() -> None:
    assert settings.ollama_generation_model == "gemma4-8k:latest"
