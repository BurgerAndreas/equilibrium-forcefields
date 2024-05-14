import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json

# define data as a json
# https://vega.github.io/vega-lite/docs/data.html
data = {
"data": {
        "values": [
            {"x": "A", "y": 28}, {"x": "B", "y": 55}, {"x": "C", "y": 43},
            {"x": "D", "y": 91}, {"x": "E", "y": 81}, {"x": "F", "y": 53},
            {"x": "G", "y": 19}, {"x": "H", "y": 87}, {"x": "I", "y": 52}
        ]
    }
}