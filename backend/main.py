"""
FastAPI backend for ML Lab Builder.

Endpoints:
    GET  /api/blocks              -> full block palette (for the frontend to render)
    GET  /api/datasets/{id}/preview -> small preview of a dataset (head rows + shape)
    POST /api/pipeline/run        -> validate, compile, train, evaluate a student's pipeline

Run locally:
    uvicorn main:app --reload --port 8000
"""

from dataclasses import asdict
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from blocks_schema import ALL_BLOCKS, BLOCKS_BY_ID, DATASET_TASK_TYPE
from datasets import load_dataset
from compiler import validate_and_compile, build_sklearn_pipeline, PipelineValidationError

app = FastAPI(title="ML Lab Builder API")

# In production (Render), set ALLOWED_ORIGIN to the deployed frontend's exact
# URL, e.g. "https://buildbench-ml.onrender.com". Falls back to "*" for local
# dev so nothing breaks when running on your own machine without the env var set.
allowed_origin = os.environ.get("ALLOWED_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class NodeIn(BaseModel):
    nodeId: str
    blockId: str
    params: dict = {}


class RunPipelineRequest(BaseModel):
    nodes: list[NodeIn]
    edges: list[dict]  # kept as raw dicts; compiler reads e["from"]/e["to"] directly


# ---------------------------------------------------------------------------
# GET /api/blocks
# ---------------------------------------------------------------------------

@app.get("/api/blocks")
def get_blocks():
    """Returns the full block palette, grouped and ready for the frontend to render."""
    result = []
    for b in ALL_BLOCKS:
        entry = {
            "id": b.id,
            "label": b.label,
            "category": b.category,
            "description": b.description,
            "params": [
                {
                    "name": p.name, "label": p.label, "kind": p.kind,
                    "default": p.default, "choices": p.choices,
                    "min": p.min, "max": p.max, "step": p.step,
                }
                for p in b.params
            ],
        }
        if b.category == "data":
            entry["taskType"] = DATASET_TASK_TYPE.get(b.id)
        result.append(entry)
    return {"blocks": result}


# ---------------------------------------------------------------------------
# GET /api/datasets/{id}/preview
# ---------------------------------------------------------------------------

@app.get("/api/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: str):
    if dataset_id not in BLOCKS_BY_ID or BLOCKS_BY_ID[dataset_id].category != "data":
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset_id}")
    bundle = load_dataset(dataset_id)
    head = bundle.X.head(5).copy()
    head["target"] = bundle.y.head(5).values
    return {
        "datasetId": dataset_id,
        "taskType": bundle.task_type,
        "rowCount": len(bundle.X),
        "featureNames": bundle.feature_names,
        "targetNames": bundle.target_names,
        "previewRows": head.to_dict(orient="records"),
        "missingValueCount": int(bundle.X.isnull().sum().sum()),
    }


# ---------------------------------------------------------------------------
# POST /api/pipeline/run
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/run")
def run_pipeline(req: RunPipelineRequest):
    nodes_in = [{"nodeId": n.nodeId, "blockId": n.blockId, "params": n.params} for n in req.nodes]

    try:
        compiled = validate_and_compile(nodes_in, req.edges)
    except PipelineValidationError as e:
        # 422 signals "your pipeline shape is invalid" -- distinct from a 500 server error,
        # frontend should render this message directly to the student.
        raise HTTPException(status_code=422, detail=str(e))

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, classification_report,
        mean_squared_error, r2_score,
    )

    X, y = compiled.dataset_bundle.X, compiled.dataset_bundle.y
    test_size = compiled.split_params.get("test_size", 0.2)
    random_state = compiled.split_params.get("random_state", 42)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        pipeline = build_sklearn_pipeline(compiled)

        start = time.time()
        pipeline.fit(X_train, y_train)
        train_seconds = round(time.time() - start, 3)

        preds = pipeline.predict(X_test)
    except Exception as e:
        # Any unexpected sklearn-level failure (bad param combo, etc.) --
        # still student-facing, but flagged as unexpected so you can see it in logs.
        raise HTTPException(status_code=500, detail=f"Training failed unexpectedly: {e}")

    metrics = {}
    task_type = compiled.dataset_bundle.task_type
    metric_block_ids = {n.block_id for n in compiled.evaluation_nodes}

    if task_type == "classification":
        if "eval:accuracy" in metric_block_ids:
            metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
        if "eval:confusion_matrix" in metric_block_ids:
            cm = confusion_matrix(y_test, preds)
            metrics["confusionMatrix"] = cm.tolist()
            metrics["confusionMatrixLabels"] = (
                compiled.dataset_bundle.target_names
                if compiled.dataset_bundle.target_names
                else [str(c) for c in sorted(set(y_test))]
            )
        if "eval:precision_recall" in metric_block_ids:
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            metrics["precisionRecall"] = report
    else:
        if "eval:rmse" in metric_block_ids:
            metrics["rmse"] = round(float(mean_squared_error(y_test, preds)) ** 0.5, 4)
        if "eval:r2" in metric_block_ids:
            metrics["r2"] = round(float(r2_score(y_test, preds)), 4)
        # Predicted vs actual points -- the single most useful regression visual,
        # always included since it's cheap and doesn't depend on which metric
        # block the student picked. Capped at 300 points so the response stays
        # small on large datasets (e.g. Housing has 20k+ rows).
        import numpy as np
        actual = np.asarray(y_test)
        predicted = np.asarray(preds)
        n_points = min(300, len(actual))
        if len(actual) > n_points:
            idx = np.random.RandomState(0).choice(len(actual), n_points, replace=False)
            actual = actual[idx]
            predicted = predicted[idx]
        metrics["predictedVsActual"] = {
            "actual": [round(float(v), 4) for v in actual],
            "predicted": [round(float(v), 4) for v in predicted],
        }

    # Feature importance -- only meaningful for tree-based models (Decision Tree,
    # Random Forest) which expose feature_importances_. Linear models, KNN, and
    # SVM don't have a comparable built-in signal, so this is simply omitted for
    # those rather than faked with something misleading.
    feature_importance = None
    fitted_model = pipeline.named_steps["model"]
    if hasattr(fitted_model, "feature_importances_"):
        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            importances = fitted_model.feature_importances_
            pairs = sorted(zip(feature_names, importances), key=lambda p: p[1], reverse=True)
            feature_importance = [
                {"feature": str(name).replace("num__", "").replace("cat__", ""), "importance": round(float(val), 4)}
                for name, val in pairs[:15]  # top 15 -- keeps chart readable
            ]
        except Exception:
            feature_importance = None  # non-critical -- omit rather than fail the whole run

    return {
        "success": True,
        "taskType": task_type,
        "trainRows": len(X_train),
        "testRows": len(X_test),
        "trainSeconds": train_seconds,
        "metrics": metrics,
        "featureImportance": feature_importance,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}