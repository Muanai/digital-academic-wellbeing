import yaml
import argparse
from pathlib import Path

# from src.data import load_data
# from src.features import generate_features
# from src.models import get_model
# from src.validation import get_cv_strategy
# from src.utils import seed_everything

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # seed_everything(config['seed'])
    print(f"Starting training with config: {config['experiment_name']}")
    
    # 1. Load data
    # 2. Generate features
    # 3. Setup CV
    # 4. Train model (e.g. LightGBM, XGBoost)
    # 5. Log metrics to MLflow / W&B
    # 6. Save model artifacts to experiments/ folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)
