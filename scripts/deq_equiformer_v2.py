import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb

"""
uses equiformer/main_md17.py
"""

import deq2ff

# from deq2ff.deq_equiformer.deq_dp_md17 import deq_dot_product_attention_transformer_exp_l2_md17
# from deq2ff.deq_equiformer.deq_graph_md17 import deq_graph_attention_transformer_nonlinear_l2_md17
from deq2ff.deq_equiformer_v2.deq_equiformer_v2_oc20 import deq_equiformer_v2_oc20

# silence:
# UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`.
# Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(
    config_name="deq", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop.

    Usage:
        python deq_equiformer.py
        python deq_equiformer.py batch_size=8
        python deq_equiformer.py +machine=vector

    Usage with slurm:
        sbatch scripts/slurm_launcher.slrm deq_equiformer.py +machine=vector
    """

    # args.model_name = "deq_dot_product_attention_transformer_exp_l2_md17"
    # args.deq_mode = True
    # args.num_layers = 2 # 6 -> 1
    # args.meas_force = True

    args.output_dir = "models/md17/deq-equiformer/test1"

    from equiformer_v2.main_oc20 import Runner

    from deq2ff.logging_utils import init_wandb

    init_wandb(args)

    Runner()(args)


if __name__ == "__main__":

    # TODO try to overfit on tiny subset of data
    # args.train_size = 100

    hydra_wrapper()
