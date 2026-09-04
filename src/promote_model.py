import dagshub
import mlflow


# ==========================================
# CONFIGURATION
# ==========================================

REGISTERED_MODEL_NAME = "SpamMessageClassifier"

DAGSHUB_OWNER = "adhavprasanna"
DAGSHUB_REPO = "spam-classifier"

PRODUCTION_ALIAS = "production"


# ==========================================
# CONNECT TO DAGSHUB
# ==========================================

dagshub.init(
    repo_owner=DAGSHUB_OWNER,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)

print(
    "MLflow Tracking URI:",
    mlflow.get_tracking_uri()
)


# ==========================================
# MLFLOW CLIENT
# ==========================================

client = mlflow.MlflowClient()


# ==========================================
# FIND ALL REGISTERED MODEL VERSIONS
# ==========================================

versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

if not versions:
    print("No registered model versions found.")
    raise SystemExit(1)


# Sort versions numerically
versions = sorted(
    versions,
    key=lambda version: int(version.version)
)


# ==========================================
# FIND LATEST MODEL VERSION
# ==========================================

latest_version = versions[-1]

latest_version_number = latest_version.version
latest_run_id = latest_version.run_id


print()
print("==============================")
print("LATEST MODEL")
print("==============================")

print(
    "Version:",
    latest_version_number
)

print(
    "Run ID:",
    latest_run_id
)


# ==========================================
# GET LATEST MODEL METRICS
# ==========================================

latest_run = client.get_run(
    latest_run_id
)

latest_f1 = latest_run.data.metrics.get(
    "f1_score"
)

if latest_f1 is None:
    print(
        "F1 score was not found for the latest model."
    )
    raise SystemExit(1)


print(
    "F1 Score:",
    latest_f1
)


# ==========================================
# CHECK CURRENT PRODUCTION MODEL
# ==========================================

try:

    production_version = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME,
        PRODUCTION_ALIAS
    )

except Exception:

    production_version = None


# ==========================================
# NO PRODUCTION MODEL
# ==========================================

if production_version is None:

    print()
    print(
        "No production model found."
    )

    print(
        f"Promoting Version {latest_version_number} "
        "to production..."
    )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        PRODUCTION_ALIAS,
        latest_version_number
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        latest_version_number,
        "promotion_status",
        "promoted"
    )

    print(
        f"Version {latest_version_number} "
        "is now production."
    )

    raise SystemExit(0)


# ==========================================
# GET PRODUCTION MODEL METRICS
# ==========================================

production_run = client.get_run(
    production_version.run_id
)

production_f1 = production_run.data.metrics.get(
    "f1_score"
)

if production_f1 is None:
    print(
        "F1 score was not found for the "
        "current production model."
    )
    raise SystemExit(1)


print()
print("==============================")
print("PRODUCTION MODEL")
print("==============================")

print(
    "Version:",
    production_version.version
)

print(
    "F1 Score:",
    production_f1
)


# ==========================================
# COMPARE MODELS
# ==========================================

print()
print("==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    f"Production F1: {production_f1:.4f}"
)

print(
    f"Candidate F1 : {latest_f1:.4f}"
)


# ==========================================
# SAME VERSION
# ==========================================

if (
    latest_version_number
    == production_version.version
):

    print()
    print(
        "Latest version is already production."
    )

    raise SystemExit(0)


# ==========================================
# PROMOTE BETTER MODEL
# ==========================================

if latest_f1 > production_f1:

    print()
    print(
        f"Candidate Version {latest_version_number} "
        "is better."
    )

    print(
        f"Promoting Version {latest_version_number} "
        "to production..."
    )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        PRODUCTION_ALIAS,
        latest_version_number
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        latest_version_number,
        "promotion_status",
        "promoted"
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        production_version.version,
        "promotion_status",
        "previous_production"
    )

    print()
    print(
        f"Version {latest_version_number} "
        "is now production."
    )


# ==========================================
# REJECT WORSE MODEL
# ==========================================

else:

    print()
    print(
        f"Candidate Version {latest_version_number} "
        "is not better than production."
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        latest_version_number,
        "promotion_status",
        "rejected"
    )

    print(
        f"Version {latest_version_number} "
        "was rejected."
    )

    print(
        f"Production remains Version "
        f"{production_version.version}."
    )