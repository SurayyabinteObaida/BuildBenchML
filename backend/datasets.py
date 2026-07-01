"""
Loads the curated datasets into a consistent shape:
    load_dataset(dataset_id) -> DatasetBundle(X: DataFrame, y: Series, feature_names, task_type)

All datasets are bundled as static CSVs under data/ -- no network calls at runtime,
so the app never breaks mid-lecture because of an external fetch failing.

X is always a pandas DataFrame (mixed numeric/categorical columns allowed) so that
preprocessing blocks like OneHotEncoder / SimpleImputer have real work to do.
Categorical columns are left as-is (object dtype); it's the student's job to add
a One-Hot Encode block if the model needs it -- that's the pedagogical point.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from blocks_schema import DATASET_TASK_TYPE

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]
    task_type: str  # "classification" | "regression"
    target_names: list[str] | None = None  # class labels, for classification only


def _load_simple_csv(filename: str, dataset_id: str, target_col: str = "target") -> DatasetBundle:
    df = pd.read_csv(DATA_DIR / filename)
    y = df.pop(target_col)
    return DatasetBundle(
        X=df,
        y=y,
        feature_names=list(df.columns),
        task_type=DATASET_TASK_TYPE[dataset_id],
    )


def _load_titanic() -> DatasetBundle:
    df = pd.read_csv(DATA_DIR / "titanic.csv")
    # Drop columns that are pure identifiers / free text -- not meaningful features
    # for a beginner pipeline; Name/Ticket/Cabin need NLP-ish handling, out of scope for v1.
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    y = df.pop("Survived")
    return DatasetBundle(
        X=df,
        y=y,
        feature_names=list(df.columns),
        task_type="classification",
        target_names=["Did not survive", "Survived"],
    )


def _load_housing() -> DatasetBundle:
    df = pd.read_csv(DATA_DIR / "housing.csv")
    y = df.pop("median_house_value")
    return DatasetBundle(
        X=df,
        y=y,
        feature_names=list(df.columns),
        task_type="regression",
    )


_LOADERS = {
    "data:iris": lambda: _load_simple_csv("iris.csv", "data:iris"),
    "data:breast_cancer": lambda: _load_simple_csv("breast_cancer.csv", "data:breast_cancer"),
    "data:wine": lambda: _load_simple_csv("wine.csv", "data:wine"),
    "data:housing": _load_housing,
    "data:titanic": _load_titanic,
}


def load_dataset(dataset_id: str) -> DatasetBundle:
    if dataset_id not in _LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    return _LOADERS[dataset_id]()
