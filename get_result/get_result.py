import wandb

wandb.login()

api = wandb.Api()

# runs = api.runs("hoangggp-uet-vnu/bundle_construction_test", filters={"name": "remove cf feat"})
# for run in runs:
#     print(run.id, run.name)

# run is specified by <entity>/<project>/<run_id>
run = api.run("hoangggp-uet-vnu/bundle_construction_test/c1ea2gu7")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics.csv")