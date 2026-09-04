import mlflow


# ==========================================
# CONFIGURATION
# ==========================================

REGISTERED_MODEL_NAME = "SpamMessageClassifier"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


# ==========================================
# SET MLFLOW TRACKING SERVER
# ==========================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

client = mlflow.MlflowClient()


# ==========================================
# GET ALL REGISTERED MODEL VERSIONS
# ==========================================

versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

if not versions:
    print("No registered model versions found.")
    raise SystemExit(1)


# ==========================================
# FIND LATEST MODEL VERSION
# ==========================================

latest_version = max(
    versions,
    key=lambda version: int(version.version)
)


# ==========================================
# FIND CURRENT PRODUCTION MODEL
# ==========================================

production_versions = [
    version
    for version in versions
    if version.tags.get("environment") == "production"
]


# ==========================================
# GET F1 SCORE FOR LATEST MODEL
# ==========================================

latest_run = client.get_run(
    latest_version.run_id
)

latest_f1 = latest_run.data.metrics.get(
    "f1_score"
)

if latest_f1 is None:
    print(
        f"F1 score not found for Version {latest_version.version}."
    )
    raise SystemExit(1)


# ==========================================
# DISPLAY LATEST MODEL
# ==========================================

print("==============================")
print("AUTOMATIC MODEL EVALUATION")
print("==============================")

print(
    f"Latest Model Version : {latest_version.version}"
)

print(
    f"Latest Model F1 Score : {latest_f1:.4f}"
)


# ==========================================
# FIRST PRODUCTION MODEL
# ==========================================

if not production_versions:

    print()
    print("No production model found.")

    print(
        f"Promoting Version {latest_version.version}"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=latest_version.version,
        key="environment",
        value="production"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=latest_version.version,
        key="promotion_status",
        value="promoted"
    )

    print()
    print(
        f"Version {latest_version.version} promoted to production."
    )

    raise SystemExit(0)


# ==========================================
# CURRENT PRODUCTION MODEL
# ==========================================

production_version = max(
    production_versions,
    key=lambda version: int(version.version)
)

production_run = client.get_run(
    production_version.run_id
)

production_f1 = production_run.data.metrics.get(
    "f1_score"
)

if production_f1 is None:
    print(
        f"F1 score not found for Production Version "
        f"{production_version.version}."
    )
    raise SystemExit(1)


# ==========================================
# DISPLAY COMPARISON
# ==========================================

print()
print("==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    f"Production Version : {production_version.version}"
)

print(
    f"Production F1      : {production_f1:.4f}"
)

print(
    f"Candidate Version  : {latest_version.version}"
)

print(
    f"Candidate F1       : {latest_f1:.4f}"
)


# ==========================================
# DO NOT PROMOTE SAME VERSION
# ==========================================

if (
    latest_version.version
    == production_version.version
):

    print()
    print(
        "Candidate is already the production model."
    )

    raise SystemExit(0)


# ==========================================
# COMPARE MODELS
# ==========================================

if latest_f1 > production_f1:

    print()
    print(
        "Candidate model is better!"
    )

    print(
        f"Promoting Version {latest_version.version}..."
    )

    # Remove production status from old model
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=production_version.version,
        key="environment",
        value="previous_production"
    )

    # Promote new model
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=latest_version.version,
        key="environment",
        value="production"
    )

    # Record promotion status
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=latest_version.version,
        key="promotion_status",
        value="promoted"
    )

    print()
    print("==============================")
    print("MODEL PROMOTED")
    print("==============================")

    print(
        f"Previous Production : Version "
        f"{production_version.version}"
    )

    print(
        f"New Production      : Version "
        f"{latest_version.version}"
    )

    print(
        f"Previous F1         : {production_f1:.4f}"
    )

    print(
        f"New F1              : {latest_f1:.4f}"
    )

else:

    print()
    print(
        "Candidate model is not better."
    )

    print(
        "Keeping current production model."
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=latest_version.version,
        key="promotion_status",
        value="rejected"
    )

    print()
    print("==============================")
    print("MODEL NOT PROMOTED")
    print("==============================")

    print(
        f"Production remains : Version "
        f"{production_version.version}"
    )

    print(
        f"Production F1      : {production_f1:.4f}"
    )

    print(
        f"Candidate F1       : {latest_f1:.4f}"
    )