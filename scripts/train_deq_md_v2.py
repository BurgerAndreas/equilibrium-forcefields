import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb

"""EquiformerV2 with training loop from EquiformerV1 for MD17."""

# register models to be used with EquiformerV1 training loop (MD17)
from equiformer.nets.registry import register_model

# DEQ EquiformerV2
from deq2ff.deq_equiformer_v2.deq_equiformer_v2_oc20 import DEQ_EquiformerV2_OC20
@register_model
def deq_equiformer_v2(**kwargs):
    return DEQ_EquiformerV2_OC20(**kwargs)

# EquiformerV2
from equiformer_v2.nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
@register_model
def equiformer_v2(**kwargs):
    return EquiformerV2_OC20(**kwargs)

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