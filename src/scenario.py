import json


def load_scenario(path):
    with open(path, "r") as f:
        return json.load(f)


def get_generator_param(scenario, category, param, default):
    try:
        return scenario["synthetic_data_params"][category][param]
    except (KeyError, TypeError):
        return default


def get_system_param(scenario, category, param, default):
    try:
        if category is None:
            return scenario["system_params"][param]
        else:
            return scenario["system_params"][category][param]
    except (KeyError, TypeError):
        return default


def get_data_path(scenario, key, default):
    try:
        return scenario["data_paths"][key]
    except (KeyError, TypeError):
        return default
