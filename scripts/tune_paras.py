
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from functools import partial
import tempfile

from ray import tune, train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch

# pip install -U "ray[data,train,tune,serve]"
# pip install bayesian-optimization

from train_deq_md import train_md


def tune_hyperparameters(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    """
    Most trials will be stopped early in order to avoid wasting resources
    https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
    https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/30bcc2970bf630097b13789b5cdcea48/hyperparameter_tuning_tutorial.ipynb
    """
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    config = {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    # a Trial Schedulers for early stopping of trials
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_acc = test_accuracy(best_trained_model, device)
        print("Best trial test set accuracy: {}".format(test_acc))
    
    return

def tune_hyperparameters_baysianopt(args):
    """
    https://docs.ray.io/en/latest/tune/examples/includes/bayesopt_example.html
    
    """

    args_tune = args.get("tune", {})
    days_per_run = args_tune.get("days_per_run", 1)
    days = args_tune.get("days", 10)
    concurrent = args_tune.get("concurrent", 1)

    #  number of hyperparameter combinations that will be tried out
    num_samples = (days // days_per_run) * concurrent

    algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
    algo = ConcurrencyLimiter(algo, max_concurrent=concurrent)

    search_space = {
        "steps": 100,
        "width": tune.uniform(0, 20),
        "height": tune.uniform(-100, 100),
        "lr": tune.loguniform(1e-4, 1e-1),
        "solver": tune.choice(["anderson", "broyden"]),
        "ln": tune.choice(["pre", "post"]),
        # "deq_kwargs.f_tol": 
    }

    # https://docs.ray.io/en/latest/tune/index.html
    # objective = training loop
    # just add the following line to the training loop:
    # tune.report(mean_loss=loss) # or # train.report({"mean_accuracy": acc})
    tuner = tune.Tuner(
        train_md,
        tune_config=tune.TuneConfig(
            metric="test_fmae",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
        ),
        # run_config=train.RunConfig(
        #     stop={"training_iteration": 5},
        # ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best hyperparameters found were: ", results.best_config)
    return


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    tune_hyperparameters(num_samples=10, max_num_epochs=10, gpus_per_trial=0)