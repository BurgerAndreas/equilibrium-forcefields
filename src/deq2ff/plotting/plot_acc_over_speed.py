import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
import copy
import os, sys, pathlib
import yaml
import json
import requests

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder, acclabels, timelabels

""" Options """
filter_eval_batch_size = 4 # 1 or 4
filter_fpreuseftol = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
time_metric = "time_forward_per_batch_test" + "_lowest" # time_test, time_forward_per_batch_test, time_forward_total_test
target = "aspirin" # aspirin, all
acc_metric = "test_f_mae_lowest" # test_f_mae_lowest, test_f_mae, test_e_mae_lowest, test_e_mae, best_test_f_mae, best_test_e_mae
layers_deq = [1, 2]
layers_equi = [1, 4, 8]
runs_with_dropout = False
# hosts = ["tacozoid11", "tacozoid10", "andreasb-lenovo"]
hosts = ["tacozoid11", "tacozoid10"]
# hosts = ["andreasb-lenovo"]

# download data or load from file
download_data = False

# choose from
eval_batch_sizes = [1, 4]
time_metrics = ["time_test", "time_forward_per_batch_test", "time_forward_total_test"]
acc_metrics = ["best_test_f_mae", "test_f_mae", "best_test_e_mae", "test_e_mae"]
acclabels.update({f"{k}_lowest": v for k, v in acclabels.items()})

