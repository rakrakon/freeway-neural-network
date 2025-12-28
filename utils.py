from pathlib import Path
import yaml

def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Project root not found")

CONFIG_PATH = "config/config.yaml"

def load_config():
    project_root = get_project_root()
    config_path = project_root / CONFIG_PATH
    with open(config_path, "r") as file:
        return yaml.safe_load(file)