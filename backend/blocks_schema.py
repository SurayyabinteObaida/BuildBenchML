"""
Fixed block vocabulary for the drag-and-drop ML pipeline builder.

Design principles:
- Every block maps to exactly one sklearn object (or a well-defined dataset/eval action).
- No free-text code, no arbitrary parameters beyond a small whitelisted set per block.
- Hyperparameters are exposed as bounded dropdowns/sliders in the UI, never open text fields.
- This file is the single source of truth for both the UI palette and the compiler.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

BlockCategory = Literal["data", "split", "preprocessing", "model", "evaluation"]


@dataclass
class ParamSpec:
    """Describes one tunable, bounded hyperparameter exposed in the UI."""
    name: str
    label: str
    kind: Literal["choice", "int", "float"]
    default: Any
    choices: list[Any] | None = None   # for kind="choice"
    min: float | None = None           # for kind="int"/"float"
    max: float | None = None
    step: float | None = None


@dataclass
class BlockDef:
    id: str
    label: str
    category: BlockCategory
    description: str
    # sklearn import path resolved at compile time, e.g. "sklearn.preprocessing.StandardScaler"
    sklearn_class: str | None
    params: list[ParamSpec] = field(default_factory=list)
    # which pipeline "slots" this block can legally connect from/to, used for graph validation
    accepts: list[BlockCategory] = field(default_factory=list)
    produces: BlockCategory | None = None


# ---------------------------------------------------------------------------
# DATA blocks — curated, pre-vetted datasets only (no arbitrary upload in v1)
# ---------------------------------------------------------------------------

DATA_BLOCKS = [
    BlockDef(
        id="data:iris",
        label="Iris Flowers",
        category="data",
        description="150 flowers, 4 measurements, 3 species. Classic classification starter.",
        sklearn_class="sklearn.datasets.load_iris",
        produces="data",
    ),
    BlockDef(
        id="data:titanic",
        label="Titanic Survival",
        category="data",
        description="Passenger records, predict who survived. Mix of numeric + categorical.",
        sklearn_class=None,  # loaded from bundled CSV, see datasets.py
        produces="data",
    ),
    BlockDef(
        id="data:housing",
        label="California Housing Prices",
        category="data",
        description="Predict median house value from district features. Regression.",
        sklearn_class="sklearn.datasets.fetch_california_housing",
        produces="data",
    ),
    BlockDef(
        id="data:breast_cancer",
        label="Breast Cancer Diagnosis",
        category="data",
        description="Tumor measurements, benign vs malignant. Binary classification.",
        sklearn_class="sklearn.datasets.load_breast_cancer",
        produces="data",
    ),
    BlockDef(
        id="data:wine",
        label="Wine Quality",
        category="data",
        description="Chemical properties of wine, predict the cultivar. Multi-class.",
        sklearn_class="sklearn.datasets.load_wine",
        produces="data",
    ),
]

# ---------------------------------------------------------------------------
# SPLIT block — exactly one required per pipeline, right after data
# ---------------------------------------------------------------------------

SPLIT_BLOCKS = [
    BlockDef(
        id="split:train_test",
        label="Train / Test Split",
        category="split",
        description="Hold out a portion of the data to test the model fairly.",
        sklearn_class="sklearn.model_selection.train_test_split",
        params=[
            ParamSpec(
                name="test_size", label="Test set size", kind="choice",
                default=0.2, choices=[0.1, 0.2, 0.3],
            ),
            ParamSpec(
                name="random_state", label="Random seed", kind="int",
                default=42, min=0, max=9999,
            ),
        ],
        accepts=["data"],
        produces="split",
    ),
]

# ---------------------------------------------------------------------------
# PREPROCESSING blocks — chainable, zero or more per pipeline
# ---------------------------------------------------------------------------

PREPROCESSING_BLOCKS = [
    BlockDef(
        id="prep:standard_scale",
        label="Standardize Features",
        category="preprocessing",
        description="Rescales numeric features to mean 0, std 1. Helps distance-based models.",
        sklearn_class="sklearn.preprocessing.StandardScaler",
        accepts=["split", "preprocessing"],
        produces="preprocessing",
    ),
    BlockDef(
        id="prep:minmax_scale",
        label="Min-Max Scale",
        category="preprocessing",
        description="Rescales numeric features into a 0-1 range.",
        sklearn_class="sklearn.preprocessing.MinMaxScaler",
        accepts=["split", "preprocessing"],
        produces="preprocessing",
    ),
    BlockDef(
        id="prep:onehot_encode",
        label="One-Hot Encode Categories",
        category="preprocessing",
        description="Turns category columns (like 'city') into numeric columns a model can use.",
        sklearn_class="sklearn.preprocessing.OneHotEncoder",
        params=[
            ParamSpec(
                name="handle_unknown", label="Unknown category handling", kind="choice",
                default="ignore", choices=["ignore", "error"],
            ),
        ],
        accepts=["split", "preprocessing"],
        produces="preprocessing",
    ),
    BlockDef(
        id="prep:impute_missing",
        label="Fill Missing Values",
        category="preprocessing",
        description="Fills gaps in the data (e.g. missing age) using the column average.",
        sklearn_class="sklearn.impute.SimpleImputer",
        params=[
            ParamSpec(
                name="strategy", label="Fill strategy", kind="choice",
                default="mean", choices=["mean", "median", "most_frequent"],
            ),
        ],
        accepts=["split", "preprocessing"],
        produces="preprocessing",
    ),
]

# ---------------------------------------------------------------------------
# MODEL blocks — exactly one required per pipeline
# ---------------------------------------------------------------------------

MODEL_BLOCKS = [
    BlockDef(
        id="model:logistic_regression",
        label="Logistic Regression",
        category="model",
        description="A simple, interpretable classifier. Good baseline for classification.",
        sklearn_class="sklearn.linear_model.LogisticRegression",
        params=[
            ParamSpec(name="C", label="Regularization strength", kind="float",
                       default=1.0, min=0.01, max=10.0, step=0.01),
            ParamSpec(name="max_iter", label="Max iterations", kind="int",
                       default=200, min=50, max=1000, step=50),
        ],
        accepts=["split", "preprocessing"],
        produces="model",
    ),
    BlockDef(
        id="model:decision_tree",
        label="Decision Tree",
        category="model",
        description="Splits data into branches based on feature thresholds. Easy to visualize.",
        sklearn_class="sklearn.tree.DecisionTreeClassifier",
        params=[
            ParamSpec(name="max_depth", label="Max depth", kind="int",
                       default=5, min=1, max=20, step=1),
        ],
        accepts=["split", "preprocessing"],
        produces="model",
    ),
    BlockDef(
        id="model:random_forest",
        label="Random Forest",
        category="model",
        description="An ensemble of decision trees voting together. Usually more accurate.",
        sklearn_class="sklearn.ensemble.RandomForestClassifier",
        params=[
            ParamSpec(name="n_estimators", label="Number of trees", kind="int",
                       default=100, min=10, max=300, step=10),
            ParamSpec(name="max_depth", label="Max depth", kind="int",
                       default=10, min=1, max=30, step=1),
        ],
        accepts=["split", "preprocessing"],
        produces="model",
    ),
    BlockDef(
        id="model:knn",
        label="K-Nearest Neighbors",
        category="model",
        description="Classifies a point based on its closest neighbors in the data.",
        sklearn_class="sklearn.neighbors.KNeighborsClassifier",
        params=[
            ParamSpec(name="n_neighbors", label="Number of neighbors (k)", kind="int",
                       default=5, min=1, max=20, step=1),
        ],
        accepts=["split", "preprocessing"],
        produces="model",
    ),
    BlockDef(
        id="model:svm",
        label="Support Vector Machine",
        category="model",
        description="Finds the best boundary line/curve to separate classes.",
        sklearn_class="sklearn.svm.SVC",
        params=[
            ParamSpec(name="kernel", label="Kernel type", kind="choice",
                       default="rbf", choices=["linear", "rbf", "poly"]),
            ParamSpec(name="C", label="Regularization strength", kind="float",
                       default=1.0, min=0.01, max=10.0, step=0.01),
        ],
        accepts=["split", "preprocessing"],
        produces="model",
    ),
    BlockDef(
        id="model:linear_regression",
        label="Linear Regression",
        category="model",
        description="Fits a straight line/plane to predict a numeric value (e.g. price).",
        sklearn_class="sklearn.linear_model.LinearRegression",
        accepts=["split", "preprocessing"],
        produces="model",
    ),
]

# ---------------------------------------------------------------------------
# EVALUATION blocks — one or more required per pipeline
# ---------------------------------------------------------------------------

EVALUATION_BLOCKS = [
    BlockDef(
        id="eval:accuracy",
        label="Accuracy Score",
        category="evaluation",
        description="Percentage of predictions the model got right. Classification only.",
        sklearn_class="sklearn.metrics.accuracy_score",
        accepts=["model"],
        produces="evaluation",
    ),
    BlockDef(
        id="eval:confusion_matrix",
        label="Confusion Matrix",
        category="evaluation",
        description="A table showing what the model predicted vs what was actually true.",
        sklearn_class="sklearn.metrics.confusion_matrix",
        accepts=["model"],
        produces="evaluation",
    ),
    BlockDef(
        id="eval:precision_recall",
        label="Precision & Recall",
        category="evaluation",
        description="How often 'positive' predictions were right, and how many positives were caught.",
        sklearn_class="sklearn.metrics.classification_report",
        accepts=["model"],
        produces="evaluation",
    ),
    BlockDef(
        id="eval:rmse",
        label="Root Mean Squared Error",
        category="evaluation",
        description="Average prediction error size, in the same units as the target. Regression only.",
        sklearn_class="sklearn.metrics.mean_squared_error",
        accepts=["model"],
        produces="evaluation",
    ),
    BlockDef(
        id="eval:r2",
        label="R² Score",
        category="evaluation",
        description="How much of the variation in the target the model explains. Regression only.",
        sklearn_class="sklearn.metrics.r2_score",
        accepts=["model"],
        produces="evaluation",
    ),
]

ALL_BLOCKS: list[BlockDef] = (
    DATA_BLOCKS + SPLIT_BLOCKS + PREPROCESSING_BLOCKS + MODEL_BLOCKS + EVALUATION_BLOCKS
)

BLOCKS_BY_ID: dict[str, BlockDef] = {b.id: b for b in ALL_BLOCKS}

# Which model blocks are valid for which task type — used to validate + to
# filter the palette once a student picks a dataset with a known task type.
CLASSIFICATION_MODELS = {
    "model:logistic_regression", "model:decision_tree", "model:random_forest",
    "model:knn", "model:svm",
}
REGRESSION_MODELS = {"model:linear_regression"}

CLASSIFICATION_METRICS = {"eval:accuracy", "eval:confusion_matrix", "eval:precision_recall"}
REGRESSION_METRICS = {"eval:rmse", "eval:r2"}

DATASET_TASK_TYPE = {
    "data:iris": "classification",
    "data:titanic": "classification",
    "data:housing": "regression",
    "data:breast_cancer": "classification",
    "data:wine": "classification",
}
