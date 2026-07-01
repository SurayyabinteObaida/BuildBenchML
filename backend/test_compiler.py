"""
Manual test suite for compiler.py -- covers happy paths across dataset types
and the key validation error cases students will hit.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from compiler import validate_and_compile, build_sklearn_pipeline, PipelineValidationError


def run_case(name, nodes, edges, expect_error=None):
    print(f"\n=== {name} ===")
    try:
        compiled = validate_and_compile(nodes, edges)
    except PipelineValidationError as e:
        if expect_error:
            assert expect_error in str(e), f"Expected error containing '{expect_error}', got: {e}"
            print(f"OK (expected error): {e}")
            return
        else:
            print(f"UNEXPECTED ERROR: {e}")
            raise
    if expect_error:
        raise AssertionError(f"Expected an error containing '{expect_error}' but compilation succeeded")

    pipeline = build_sklearn_pipeline(compiled)
    X, y = compiled.dataset_bundle.X, compiled.dataset_bundle.y
    test_size = compiled.split_params.get("test_size", 0.2)
    random_state = compiled.split_params.get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    if compiled.dataset_bundle.task_type == "classification":
        print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    else:
        print("RMSE:", round(mean_squared_error(y_test, preds) ** 0.5, 2))
        print("R2:", round(r2_score(y_test, preds), 4))
    print("Compiled OK, pipeline trained and evaluated successfully.")


# --- Happy path: Iris, clean numeric data, minimal pipeline ---
run_case(
    "Iris - minimal pipeline",
    nodes=[
        {"nodeId": "n1", "blockId": "data:iris"},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {"test_size": 0.2, "random_state": 42}},
        {"nodeId": "n3", "blockId": "model:logistic_regression", "params": {"C": 1.0, "max_iter": 200}},
        {"nodeId": "n4", "blockId": "eval:accuracy"},
    ],
    edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n3", "to": "n4"}],
)

# --- Happy path: Titanic, needs impute + onehot ---
run_case(
    "Titanic - with preprocessing chain",
    nodes=[
        {"nodeId": "n1", "blockId": "data:titanic"},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {"test_size": 0.2, "random_state": 42}},
        {"nodeId": "n3", "blockId": "prep:impute_missing", "params": {"strategy": "median"}},
        {"nodeId": "n4", "blockId": "prep:onehot_encode", "params": {"handle_unknown": "ignore"}},
        {"nodeId": "n5", "blockId": "model:random_forest", "params": {"n_estimators": 100, "max_depth": 10}},
        {"nodeId": "n6", "blockId": "eval:accuracy"},
        {"nodeId": "n7", "blockId": "eval:confusion_matrix"},
    ],
    edges=[
        {"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n3", "to": "n4"},
        {"from": "n4", "to": "n5"}, {"from": "n5", "to": "n6"}, {"from": "n6", "to": "n7"},
    ],
)

# --- Happy path: Titanic WITHOUT preprocessing blocks (tests auto-fallback doesn't crash) ---
run_case(
    "Titanic - no preprocessing blocks (auto-fallback)",
    nodes=[
        {"nodeId": "n1", "blockId": "data:titanic"},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {}},
        {"nodeId": "n3", "blockId": "model:decision_tree", "params": {"max_depth": 5}},
        {"nodeId": "n4", "blockId": "eval:accuracy"},
    ],
    edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n3", "to": "n4"}],
)

# --- Happy path: Housing, regression ---
run_case(
    "Housing - regression pipeline",
    nodes=[
        {"nodeId": "n1", "blockId": "data:housing"},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {"test_size": 0.2}},
        {"nodeId": "n3", "blockId": "prep:impute_missing", "params": {"strategy": "median"}},
        {"nodeId": "n4", "blockId": "prep:onehot_encode"},
        {"nodeId": "n5", "blockId": "prep:standard_scale"},
        {"nodeId": "n6", "blockId": "model:linear_regression"},
        {"nodeId": "n7", "blockId": "eval:rmse"},
        {"nodeId": "n8", "blockId": "eval:r2"},
    ],
    edges=[
        {"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n3", "to": "n4"},
        {"from": "n4", "to": "n5"}, {"from": "n5", "to": "n6"}, {"from": "n6", "to": "n7"},
        {"from": "n7", "to": "n8"},
    ],
)

# --- Error case: empty canvas ---
run_case("Error - empty canvas", nodes=[], edges=[], expect_error="canvas is empty")

# --- Error case: missing split block ---
run_case(
    "Error - missing split",
    nodes=[
        {"nodeId": "n1", "blockId": "data:iris"},
        {"nodeId": "n2", "blockId": "model:logistic_regression"},
        {"nodeId": "n3", "blockId": "eval:accuracy"},
    ],
    edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}],
    expect_error="Train/Test Split",
)

# --- Error case: wrong model type for task (regression model on classification data) ---
run_case(
    "Error - regression model on classification dataset",
    nodes=[
        {"nodeId": "n1", "blockId": "data:iris"},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {}},
        {"nodeId": "n3", "blockId": "model:linear_regression"},
        {"nodeId": "n4", "blockId": "eval:accuracy"},
    ],
    edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n3", "to": "n4"}],
    expect_error="regression model",
)

# --- Error case: wrong metric type ---
run_case(
    "Error - RMSE metric on classification dataset",
    nodes=[
        {"nodeId": "n1", "blockId": "data:iris"},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {}},
        {"nodeId": "n3", "blockId": "model:logistic_regression"},
        {"nodeId": "n4", "blockId": "eval:rmse"},
    ],
    edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n3", "to": "n4"}],
    expect_error="regression metric",
)

# --- Error case: no evaluation block ---
run_case(
    "Error - no evaluation block",
    nodes=[
        {"nodeId": "n1", "blockId": "data:iris"},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {}},
        {"nodeId": "n3", "blockId": "model:logistic_regression"},
    ],
    edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}],
    expect_error="evaluation block",
)

# --- Error case: branching (two outgoing edges from one node) ---
run_case(
    "Error - branching not yet supported",
    nodes=[
        {"nodeId": "n1", "blockId": "data:iris"},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {}},
        {"nodeId": "n3", "blockId": "model:logistic_regression"},
        {"nodeId": "n4", "blockId": "model:random_forest"},
        {"nodeId": "n5", "blockId": "eval:accuracy"},
    ],
    edges=[
        {"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n2", "to": "n4"},
        {"from": "n3", "to": "n5"},
    ],
    expect_error="more than one next step",
)

# --- Error case: disconnected block ---
run_case(
    "Error - disconnected block",
    nodes=[
        {"nodeId": "n1", "blockId": "data:iris"},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {}},
        {"nodeId": "n3", "blockId": "model:logistic_regression"},
        {"nodeId": "n4", "blockId": "eval:accuracy"},
        {"nodeId": "n5", "blockId": "prep:standard_scale"},  # floating, unconnected
    ],
    edges=[{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n3", "to": "n4"}],
    expect_error="disconnected chain",
)

print("\n\nALL TESTS PASSED")
