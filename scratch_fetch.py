import sqlite3
import json

conn = sqlite3.connect('mlflow.db')
cur = conn.cursor()

cur.execute("SELECT experiment_id FROM experiments WHERE name = 'Phase17_Deep_Tuning'")
exp_id = cur.fetchone()[0]

best_params_all = {}

for model in ['LightGBM', 'XGBoost']:
    run_name = f'DeepTuning_{model}'
    cur.execute('''
        SELECT run_uuid FROM runs 
        WHERE experiment_id = ? AND name = ?
        ORDER BY start_time DESC LIMIT 1
    ''', (exp_id, run_name))
    run_uuid = cur.fetchone()[0]
    
    cur.execute("SELECT value FROM metrics WHERE run_uuid = ? AND key = 'best_macro_f1'", (run_uuid,))
    best_f1 = cur.fetchone()[0]
    
    cur.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_uuid,))
    params = {}
    for row in cur.fetchall():
        key, val = row
        try:
            if '.' in val or 'e' in val.lower():
                params[key] = float(val)
            else:
                params[key] = int(val)
        except:
            params[key] = val
            
    best_params_all[model] = {'best_macro_f1': best_f1, 'params': params}

with open('experiments/phase17_best_params.json', 'w') as f:
    json.dump(best_params_all, f, indent=4)

print('Saved successfully to phase17_best_params.json')
