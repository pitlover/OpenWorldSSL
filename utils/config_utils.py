from argparse import ArgumentParser, REMAINDER
from typing import List, Dict
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "prepare_config",
    "default_parser",
    "load_config",
    "override_config_by_cli",
    "resolve_config",
]


def prepare_config():
    """All-in-one common workflow."""
    parser = default_parser()
    args = parser.parse_args()
    # register_custom_resolvers()
    d_config = load_config(args.config)
    d_config = override_config_by_cli(d_config, args.script_args)
    config = resolve_config(d_config)
    return args, config


def default_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML configuration", required=True)
    parser.add_argument("--debug", action="store_true", help="Debugging flag")
    parser.add_argument("--eval", action="store_true", help="Testing flag")
    parser.add_argument("script_args", nargs=REMAINDER, help="Override config by CLI")
    return parser


def register_custom_resolvers() -> None:
    """Register custom resolvers for OmegaConf.

    Note:
        Do not recommend registering general operations, such as 'eval'.
        They tend to cause many exceptions...

    Usages:
        "${add:${epoch},${additional},10}"
        "${sub:${batch_size},16}"
        "${mul:${batch_size},${lr}}"
        "${div:${lr},${batch_size}}"
        "${cond:${fp16},16,32}"
        "${idiv:${batch_size},${gpus}}"
        "${idiv2:${batch_size},${num_accum},${gpus}}"
    """
    OmegaConf.register_resolver("add", lambda *x: sum(x))
    OmegaConf.register_resolver("sub", lambda x, y: x - y)
    OmegaConf.register_resolver("mul", lambda x, y: x * y)
    OmegaConf.register_resolver("div", lambda x, y: float(x / y))
    OmegaConf.register_resolver("cond", lambda c, y, n: y if c else n)
    OmegaConf.register_resolver("idiv", lambda x, y: max(x // y, 1))
    OmegaConf.register_resolver("idiv2", lambda x, y, z: max(max(x // y, 1) // z, 1))
    OmegaConf.register_resolver("max", lambda x, y: max(x, y))
    OmegaConf.register_resolver("min", lambda x, y: min(x, y))
    OmegaConf.register_resolver("clamp", lambda x, a, b: max(min(x, b), a))


def load_config(yaml_path: str) -> DictConfig:
    cfg = OmegaConf.load(yaml_path)
    return cfg


def override_config_by_cli(base_cfg: DictConfig, script_args: List[str]) -> DictConfig:
    """Override the read config config.

    Note:
        See OmegaConf for details.
        Usage of CLI script_args:
            wandb.mode=disabled
            adam.betas="[0.9, 0.98]"
            resume.ignore_keys="['optimizer', 'scheduler']"
    """
    cli_cfg = OmegaConf.from_dotlist(script_args)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    return cfg


def resolve_config(cfg: DictConfig) -> Dict:
    return OmegaConf.to_container(cfg, resolve=True)