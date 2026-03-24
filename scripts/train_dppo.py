"""
Launch DPPO diffusion policy pre-training via external/dsrl/dppo
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DPPO_ROOT = PROJECT_ROOT / "external" / "dsrl" / "dppo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Training DPPO diffusion policy")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Hydra YAML training config",
    )
    parser.add_argument(
        "--dppo_repo",
        type=str,
        default=str(DPPO_ROOT),
        help="Path to the DPPO repository (external/dsrl/dppo by default)",
    )
    args = parser.parse_args()

    # Ensure DPPO repo is on the Python path
    dppo_root = Path(args.dppo_repo).resolve()
    if not dppo_root.exists():
        raise FileNotFoundError(
            f"DPPO repo not found at {dppo_root}. "
            "Clone your fork into external/dsrl/dppo or specify --dppo_repo."
        )
    dppo_str = str(dppo_root)
    if dppo_str not in sys.path:
        sys.path.insert(0, dppo_str)

    # Register OmegaConf resolvers before Hydra compose
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)

    # Compose config via Hydra
    from hydra import compose, initialize_config_dir
    import hydra

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with initialize_config_dir(
        config_dir=str(config_path.parent),
        version_base=None,
    ):
        cfg = compose(config_name=config_path.stem)

    # Resolve all interpolations eagerly (so ${now:} timestamps are consistent)
    OmegaConf.resolve(cfg)

    logger.info("Config loaded from %s", config_path)
    logger.info("Instantiating training agent: %s", cfg.get("_target_", "unknown"))

    # The DPPO agent expects the entire config as a single positional arg,
    # so we use get_class + direct construction (matching dppo/script/run.py).
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
