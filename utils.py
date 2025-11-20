import yaml

CONFIG_PATH = "config/config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)