import mlflow

# ==========================================
# CONFIGURATION
# ==========================================

REGISTERED_MODEL_NAME = "SpamMessageClassifier"
MODEL_VERSION = 2


# ==========================================
# SET MLFLOW TRACKING SERVER
# ==========================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)


# ==========================================
# PROMOTE MODEL VERSION
# ==========================================

client = mlflow.MlflowClient()

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=MODEL_VERSION,
    key="environment",
    value="production"
)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("==============================")
print("MODEL PROMOTION")
print("==============================")

print(
    f"Model: {REGISTERED_MODEL_NAME}"
)

print(
    f"Version: {MODEL_VERSION}"
)

print(
    "Environment: production"
)

print("\nModel promoted successfully!")