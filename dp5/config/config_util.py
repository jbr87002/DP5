from pathlib import Path

def convert_paths_to_strings(config):
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
        elif isinstance(value, list):
            config[key] = [str(v) if isinstance(v, Path) else v for v in value]
        elif isinstance(value, dict):
            convert_paths_to_strings(value)