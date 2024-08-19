# Copyright (c) 2024, Aviv Bick, Kevin Li.

import json
from collections.abc import Mapping

import yaml


class CustomDumper(yaml.Dumper):
    """Custom YAML dumper"""

    def increase_indent(self, flow=False, indentless=False):
        return super(CustomDumper, self).increase_indent(
            flow=flow, indentless=indentless
        )


def dict_representer(dumper, data):
    """Custom representer for dictionaries"""

    def is_last_tier(d):
        return all(not isinstance(v, dict) for v in d.values())

    if is_last_tier(data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)
    else:
        return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=False)


def list_representer(dumper, data):
    """Custom representer for lists"""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


class Config(Mapping):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    # Support item assignment
    def __setitem__(self, key, value):
        setattr(self, key, value)

    # Support ** unpacking
    def __iter__(self):
        return iter(self.__dict__)

    # Support in operator
    def __contains__(self, key):
        return hasattr(self, key)

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        CustomDumper.add_representer(dict, dict_representer)
        CustomDumper.add_representer(list, list_representer)
        # Print the dictionary as YAML with the specified styles and line width
        return yaml.dump(
            self.to_dict(), Dumper=CustomDumper, default_flow_style=False, width=150
        )

    def to_dict(self):
        final_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                final_dict[key] = value.to_dict()
            else:
                final_dict[key] = value
        return final_dict

    @classmethod
    def from_dict(cls, _dict):
        if not isinstance(_dict, dict):
            raise ValueError("Input data must be a dictionary")
        return setup_config(_dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r") as f:
            return cls.from_dict(yaml.load(f, Loader=yaml.FullLoader))

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_file(cls, path):
        if path.endswith(".yaml"):
            return cls.from_yaml(path)
        elif path.endswith(".json"):
            return cls.from_json(path)
        else:
            raise ValueError("Only YAML and JSON files are supported")

    def iterate_items(self, root=""):
        for key, value in self.__dict__.items():
            path = f"{root}.{key}" if root else key
            if isinstance(value, Config):
                yield from value.iterate_items(root=path)
            else:
                yield (path, value)


def setup_config(cfg):
    """
    Recursively turns the config dictionary into a dataclass
    """
    for key, value in cfg.items():
        if isinstance(value, dict):
            cfg[key] = setup_config(cfg=value)
    return Config(**cfg)


def _load_config(config_path):
    """
    Recursively load the config file and its dependencies
    """
    with open(config_path, "r") as f:
        lines = f.readlines()
    specials = [line for line in lines if line[0] == "@"]
    lines = [line for line in lines if line[0] != "@"]
    config = yaml.load("".join(lines), Loader=yaml.FullLoader)
    final_config = {}
    for special in specials:
        # if @LOAD : _path_ -> load the file (recursively)
        path = special.split(" ")[1].strip()
        loaded_config = _load_config(path)
        final_config = recursive_update(final_config, loaded_config)
    final_config = recursive_update(
        final_config, config
    )  # config elements have precedence over loaded elements
    return final_config


def load_config(config_path):
    """
    Load the config file
    """
    config = _load_config(config_path)
    return Config.from_dict(config)


def recursive_update(d, u):
    """
    Precedence: u over d
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
