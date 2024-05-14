import wandb

# Summary metrics are the last value of your logged metrics. 
# If you’re logging metrics over time/steps then you could retrieve them using our Public API with the methods history and scan_history. 
# scan_history returns the unsampled metrics (all your steps) while history returns sampled metrics 


api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
run = api.run("<entity>/<project>/<run_id>")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics.csv")


run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
