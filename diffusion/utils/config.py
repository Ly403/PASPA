import importlib
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig


def set_config_value(config, key, val):
    if key == "target":
        if isinstance(config, DictConfig):
            with open_dict(config):
                config[key] = val
        else:
            config[key] = val
    else:
        if isinstance(config, DictConfig):
            with open_dict(config):
                if "params" in config:
                    config["params"][key] = val
                else:
                    config["parmas"] = {key: val}  # FIXME: typo (kept for now to avoid behavior change)
        else:
            if "params" in config:
                config["params"][key] = val
            else:
                config["parmas"] = {key: val}  # FIXME: typo (kept for now to avoid behavior change)
    return config


def set_value(config, key, val):
    if isinstance(config, DictConfig):
        with open_dict(config):
            config[key] = val
    else:
        config[key] = val
    return config


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if isinstance(config, DictConfig):
        config = OmegaConf.to_object(config)
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
