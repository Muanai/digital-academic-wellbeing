import sqlite3
import os

def merge_db():
    if not os.path.exists('mlflow.db') or not os.path.exists('mlflow_tuning.db'):
        print("Error: Both mlflow.db and mlflow_tuning.db must exist.")
        return
    
    # Connect to main DB and attach tuning DB
    conn = sqlite3.connect('mlflow.db')
    cur = conn.cursor()
    cur.execute("ATTACH DATABASE 'mlflow_tuning.db' AS tuning")
    
    # Get the max experiment ID in mlflow.db to shift the IDs from tuning
    cur.execute("SELECT MAX(experiment_id) FROM experiments")
    max_id = cur.fetchone()[0] or 0
    offset = max_id + 1
    
    print(f"Max experiment_id in mlflow.db is {max_id}. Applying offset {offset} to mlflow_tuning.db to avoid clashes.")
    
    try:
        # 1. Merge experiments (only if name doesn't exist)
        cur.execute(f"""
            INSERT INTO experiments (experiment_id, name, artifact_location, lifecycle_stage, creation_time, last_update_time)
            SELECT experiment_id + {offset}, name, artifact_location, lifecycle_stage, creation_time, last_update_time 
            FROM tuning.experiments 
            WHERE name NOT IN (SELECT name FROM experiments)
        """)
        
        # 2. Merge runs (update experiment_id with offset)
        cur.execute(f"""
            INSERT INTO runs (run_uuid, name, source_type, source_name, entry_point_name, user_id, status, start_time, end_time, source_version, lifecycle_stage, artifact_uri, experiment_id, deleted_time)
            SELECT run_uuid, name, source_type, source_name, entry_point_name, user_id, status, start_time, end_time, source_version, lifecycle_stage, artifact_uri, experiment_id + {offset}, deleted_time
            FROM tuning.runs
            WHERE run_uuid NOT IN (SELECT run_uuid FROM runs)
        """)
        
        # 3. Merge metrics
        # (Assuming no duplicates for the exact same run_uuid/key/timestamp/step combination)
        cur.execute(f"""
            INSERT INTO metrics (key, value, timestamp, run_uuid, step, is_nan)
            SELECT key, value, timestamp, run_uuid, step, is_nan
            FROM tuning.metrics
            WHERE run_uuid IN (SELECT run_uuid FROM tuning.runs)
        """)
        
        # 4. Merge params
        cur.execute(f"""
            INSERT INTO params (key, value, run_uuid)
            SELECT key, value, run_uuid
            FROM tuning.params
            WHERE run_uuid IN (SELECT run_uuid FROM tuning.runs)
        """)
        
        # 5. Merge tags
        cur.execute(f"""
            INSERT INTO tags (key, value, run_uuid)
            SELECT key, value, run_uuid
            FROM tuning.tags
            WHERE run_uuid IN (SELECT run_uuid FROM tuning.runs)
        """)
        
        conn.commit()
        print("✅ Merge successful! All tuning runs from 'mlflow_tuning.db' have been safely copied into 'mlflow.db'.")
        print("ℹ️ You can now run 'mlflow ui' as usual to see all experiments in one place.")
    except Exception as e:
        conn.rollback()
        print("❌ Error during merge:", e)
    finally:
        conn.close()

if __name__ == '__main__':
    merge_db()
