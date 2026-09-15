import yaml
import argparse
from pathlib import Path
import pandas as pd

# from src.data import load_data
# from src.features import generate_features
# from src.models import load_model

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print(f"Starting inference with config: {config['experiment_name']}")
    
    # 1. Load test data
    # 2. Generate features (using same pipeline as train)
    # 3. Load saved models from experiments/
    # 4. Predict & Ensemble
    # 5. Save to data/submissions/submission.csv
    
    # Example snippet:
    # sub = pd.DataFrame({'id': test_ids, 'target': preds})
    # sub.to_csv("data/submissions/submission.csv", index=False)
    print("Inference completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)
