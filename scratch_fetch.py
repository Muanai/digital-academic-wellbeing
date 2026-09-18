import mlflow
import json
import os

mlflow.set_tracking_uri('sqlite:///mlflow.db')
experiment_name = 'Phase_6_Ensemble'
exp = mlflow.get_experiment_by_name(experiment_name)
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
run = runs[runs['tags.mlflow.runName'] == 'Scenario_4_TreeBased'].iloc[0]

print('--- Phase 6 Best Params ---')
for k, v in run.items():
    if 'params.weight_' in k:
        print(f'{k}: {v}')

import tempfile
client = mlflow.tracking.MlflowClient()
local_path = client.download_artifacts(run.run_id, 'ensemble_config')
for f in os.listdir(local_path):
    with open(os.path.join(local_path, f)) as fp:
        print(fp.read())
