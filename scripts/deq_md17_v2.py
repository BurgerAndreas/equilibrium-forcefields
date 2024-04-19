import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb

"""EquiformerV2 with training loop from EquiformerV1 for MD17."""

# DEQ EquiformerV2
from deq2ff.deq_equiformer_v2.deq_equiformer_v2_oc20 import (
    deq_equiformer_v2_oc20,
)

@hydra.main(
    config_name="oc20", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    from deq2ff.logging_utils import init_wandb

    init_wandb(args, project="equilibrium-forcefields-equiformer_v2")


if __name__ == "__main__":
    hydra_wrapper()