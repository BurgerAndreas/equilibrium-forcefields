import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb

"""EquiformerV2 with training loop from EquiformerV1 for MD17."""

# register all models
import deq2ff.register_all_models


@hydra.main(
    config_name="md17", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    # also load the args from equiformer/config/md17.yaml
    # and update the args with the new args (if not already present)
    # args = OmegaConf.merge(args, OmegaConf.load("equiformer/config/md17.yaml"))
    # argsmd17 = OmegaConf.load("equiformer/config/md17.yaml")
    # argsmd17.update(args)
    # args = argsmd17

    from deq2ff.logging_utils import init_wandb

    # init_wandb(args, project="equilibrium-forcefields-equiformer_v2")
    init_wandb(args)

    from train_deq_md import main

    main(args)


if __name__ == "__main__":
    hydra_wrapper()
