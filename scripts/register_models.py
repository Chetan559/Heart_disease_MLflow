import mlflow
from src.data.load_data import load_config

config = load_config("config.yaml")
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
client = mlflow.MlflowClient()

reg_models = config["mlflow"]["registered_models"]
alias      = config["mlflow"]["champion_alias"]

print("Available models:")
for key, name in reg_models.items():
    print(f"  [{key}] → {name}")

model_key = input("\nWhich model to register? (xgb / lgb / cat / meta): ").strip()
run_id    = input("Enter Run ID: ").strip()

if model_key not in reg_models:
    print(f"Unknown model key: {model_key}")
    exit(1)

model_name = reg_models[model_key]
model_uri  = f"runs:/{run_id}/model"

print(f"\nRegistering '{model_name}' from run {run_id}...")
mv = mlflow.register_model(model_uri=model_uri, name=model_name)

# Set alias so Streamlit picks it up immediately without any code change
client.set_registered_model_alias(model_name, alias, mv.version)

print(f"✅ Registered: {model_name} v{mv.version} → alias='{alias}'")
print(f"   Streamlit will now serve this version automatically.")