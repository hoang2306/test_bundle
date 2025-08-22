import pandas as pd 
import wandb 

wandb.login()
api = wandb.Api()


# metric_path = 'metrics.csv'
# metric_df = pd.read_csv(metric_path)

# print(metric_df.keys())
# print(metric_df.head())

run = api.run("hoangggp-uet-vnu/bundle_construction_test/c1ea2gu7")
# if run.state == "finished":
#     for i, row in run.history().iterrows():
#         print(row)
#         break

history = run.scan_history()
# print(history)
summary = run.summary
print(summary)