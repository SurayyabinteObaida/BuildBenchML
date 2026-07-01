"""
Compiles a student-built block graph into an actual, runnable sklearn Pipeline.

Pipeline shape enforced (v1, linear only -- no branching/ensembling yet, that's
a v2 feature once linear pipelines are solid):

    data -> split -> [preprocessing]* -> model -> [evaluation]+

Validation happens in two layers:
  1. Structural: right block types in the right order, no missing/extra pieces.
  2. Semantic: model type matches dataset task type, metric types match task type.

Both layers produce clear, student-facing error messages -- these errors ARE
part of the pedagogy, so they should say *why* something is wrong, not just that
it is.
"""

from dataclasses import dataclass
import importlib

from blocks_schema import (
    BLOCKS_BY_ID,
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
)
from datasets import load_dataset, DatasetBundle


class PipelineValidationError(Exception):
    """Raised for any student-facing validation failure. Message is shown as-is in the UI."""
    pass


@dataclass
class GraphNode:
    node_id: str
    block_id: str
    params: dict


@dataclass
class CompiledPipeline:
    dataset_bundle: DatasetBundle
    split_params: dict
    preprocessing_node_ids: list[str]      # in order
    preprocessing_blocks: list[GraphNode]
    model_node: GraphNode
    evaluation_nodes: list[GraphNode]
    sklearn_pipeline: "Pipeline"           # noqa: F821 -- built lazily, see build_sklearn_pipeline


def _resolve_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _linearize(nodes: list[GraphNode], edges: list[tuple[str, str]]) -> list[GraphNode]:
    """
    v1 only supports a single linear chain (no branching). Returns nodes in
    execution order by following edges from the node with no incoming edge.
    Raises PipelineValidationError on cycles, branches, or disconnected nodes.
    """
    if not nodes:
        raise PipelineValidationError(
            "Your canvas is empty. Start by dragging a dataset block onto the canvas."
        )

    incoming = {n.node_id: 0 for n in nodes}
    outgoing = {n.node_id: [] for n in nodes}
    for src, dst in edges:
        if src not in outgoing or dst not in incoming:
            raise PipelineValidationError("An edge connects to a block that doesn't exist.")
        outgoing[src].append(dst)
        incoming[dst] += 1

    # branching check: v1 pipelines must be a single chain
    for node_id, out_list in outgoing.items():
        if len(out_list) > 1:
            raise PipelineValidationError(
                "One of your blocks connects to more than one next step. "
                "For now, pipelines must be a single straight line -- branching pipelines "
                "are coming in a future version."
            )
    for node_id, count in incoming.items():
        if count > 1:
            raise PipelineValidationError(
                "One of your blocks has more than one incoming connection. "
                "Pipelines must be a single straight line for now."
            )

    starts = [n for n in nodes if incoming[n.node_id] == 0]
    if len(starts) == 0:
        raise PipelineValidationError("Your pipeline has a cycle -- blocks can't loop back on themselves.")
    if len(starts) > 1:
        raise PipelineValidationError(
            "Your canvas has more than one disconnected chain. Connect all your blocks into one path."
        )

    by_id = {n.node_id: n for n in nodes}
    ordered = []
    current = starts[0]
    visited = set()
    while True:
        if current.node_id in visited:
            raise PipelineValidationError("Your pipeline has a cycle -- blocks can't loop back on themselves.")
        visited.add(current.node_id)
        ordered.append(current)
        nexts = outgoing[current.node_id]
        if not nexts:
            break
        current = by_id[nexts[0]]

    if len(ordered) != len(nodes):
        raise PipelineValidationError(
            "Some blocks on your canvas aren't connected to the main pipeline. "
            "Remove them or connect them in."
        )
    return ordered


def validate_and_compile(
    nodes_in: list[dict],
    edges_in: list[dict],
) -> CompiledPipeline:
    """
    nodes_in: [{"nodeId": str, "blockId": str, "params": dict}, ...]
    edges_in: [{"from": str, "to": str}, ...]
    """
    nodes = [GraphNode(n["nodeId"], n["blockId"], n.get("params", {})) for n in nodes_in]
    edges = [(e["from"], e["to"]) for e in edges_in]

    for n in nodes:
        if n.block_id not in BLOCKS_BY_ID:
            raise PipelineValidationError(f"Unknown block: {n.block_id}")

    ordered = _linearize(nodes, edges)

    categories = [BLOCKS_BY_ID[n.block_id].category for n in ordered]

    # --- structural shape check: data, split, prep*, model, eval+
    if categories[0] != "data":
        raise PipelineValidationError("Every pipeline must start with a dataset block.")
    if len(categories) < 2 or categories[1] != "split":
        raise PipelineValidationError(
            "Add a Train/Test Split block right after your dataset. "
            "This holds out data to test the model fairly."
        )

    i = 2
    prep_nodes = []
    while i < len(categories) and categories[i] == "preprocessing":
        prep_nodes.append(ordered[i])
        i += 1

    if i >= len(categories) or categories[i] != "model":
        raise PipelineValidationError(
            "Your pipeline needs exactly one model block after preprocessing "
            "(e.g. Logistic Regression, Random Forest)."
        )
    model_node = ordered[i]
    i += 1

    eval_nodes = ordered[i:]
    if not eval_nodes:
        raise PipelineValidationError(
            "Add at least one evaluation block (e.g. Accuracy Score) after your model "
            "so you can see how well it performed."
        )
    if any(BLOCKS_BY_ID[n.block_id].category != "evaluation" for n in eval_nodes):
        raise PipelineValidationError(
            "Only evaluation blocks can come after the model block."
        )

    # --- semantic check: model / metrics must match dataset task type
    dataset_bundle = load_dataset(ordered[0].block_id)
    task_type = dataset_bundle.task_type

    model_block_id = model_node.block_id
    if task_type == "classification" and model_block_id not in CLASSIFICATION_MODELS:
        raise PipelineValidationError(
            f"'{BLOCKS_BY_ID[model_block_id].label}' is a regression model, but this dataset "
            f"needs classification (predicting a category, not a number). "
            f"Try Logistic Regression, Decision Tree, Random Forest, KNN, or SVM instead."
        )
    if task_type == "regression" and model_block_id not in REGRESSION_MODELS:
        raise PipelineValidationError(
            f"'{BLOCKS_BY_ID[model_block_id].label}' is a classification model, but this dataset "
            f"needs regression (predicting a number, not a category). "
            f"Try Linear Regression instead."
        )

    for en in eval_nodes:
        if task_type == "classification" and en.block_id not in CLASSIFICATION_METRICS:
            raise PipelineValidationError(
                f"'{BLOCKS_BY_ID[en.block_id].label}' is a regression metric, but this is a "
                f"classification problem. Try Accuracy Score, Confusion Matrix, or Precision & Recall."
            )
        if task_type == "regression" and en.block_id not in REGRESSION_METRICS:
            raise PipelineValidationError(
                f"'{BLOCKS_BY_ID[en.block_id].label}' is a classification metric, but this is a "
                f"regression problem. Try RMSE or R\u00b2 Score."
            )

    split_node = ordered[1]

    return CompiledPipeline(
        dataset_bundle=dataset_bundle,
        split_params=split_node.params,
        preprocessing_node_ids=[n.node_id for n in prep_nodes],
        preprocessing_blocks=prep_nodes,
        model_node=model_node,
        evaluation_nodes=eval_nodes,
        sklearn_pipeline=None,  # filled in by build_sklearn_pipeline, kept separate for testability
    )


def build_sklearn_pipeline(compiled: CompiledPipeline):
    """
    Builds the actual sklearn Pipeline object. Kept separate from validate_and_compile
    so validation logic can be unit tested without needing sklearn objects instantiated.

    Handles the numeric/categorical split automatically via ColumnTransformer:
    - If the student added a One-Hot Encode block, categorical columns go through it.
    - If the student added a Fill Missing Values block, it's applied to numeric columns
      (SimpleImputer(strategy=mean/median) only makes sense on numeric data; most_frequent
      is applied dataset-wide if explicitly chosen).
    - Any other preprocessing block (scalers) applies to numeric columns only.

    This auto-routing keeps the block model simple for students (one block = one concept)
    while still producing a correct, runnable pipeline under the hood.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    X = compiled.dataset_bundle.X
    numeric_cols = list(X.select_dtypes(include="number").columns)
    categorical_cols = list(X.select_dtypes(exclude="number").columns)

    numeric_steps = []
    categorical_steps = []

    for node in compiled.preprocessing_blocks:
        block = BLOCKS_BY_ID[node.block_id]
        cls = _resolve_class(block.sklearn_class)
        kwargs = {p.name: node.params.get(p.name, p.default) for p in block.params}

        if block.id == "prep:onehot_encode":
            categorical_steps.append((node.node_id, cls(**kwargs)))
        elif block.id == "prep:impute_missing":
            numeric_steps.append((node.node_id, cls(**kwargs)))
            if categorical_cols:
                categorical_steps.append(
                    (node.node_id + "_cat", SimpleImputer(strategy="most_frequent"))
                )
        else:  # scalers etc. -- numeric only
            numeric_steps.append((node.node_id, cls(**kwargs)))

    # categorical columns always need at least a fallback encode+impute so the
    # model doesn't choke on string columns, even if the student forgot the block --
    # this is intentionally forgiving so a missed block produces a worse score,
    # not a crash.
    if categorical_cols and not any(
        n.block_id == "prep:onehot_encode" for n in compiled.preprocessing_blocks
    ):
        categorical_steps.append(("auto_impute", SimpleImputer(strategy="most_frequent")))
        categorical_steps.append(("auto_encode", OneHotEncoder(handle_unknown="ignore")))
    if categorical_cols and not any(
        n.block_id == "prep:impute_missing" for n in compiled.preprocessing_blocks
    ):
        categorical_steps.insert(0, ("auto_impute_cat", SimpleImputer(strategy="most_frequent")))

    if numeric_cols and not any(
        n.block_id == "prep:impute_missing" for n in compiled.preprocessing_blocks
    ) and X[numeric_cols].isnull().any().any():
        numeric_steps.insert(0, ("auto_impute_num", SimpleImputer(strategy="mean")))

    numeric_pipeline = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    categorical_pipeline = Pipeline(categorical_steps) if categorical_steps else "drop"

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers)

    model_block = BLOCKS_BY_ID[compiled.model_node.block_id]
    model_cls = _resolve_class(model_block.sklearn_class)
    model_kwargs = {
        p.name: compiled.model_node.params.get(p.name, p.default) for p in model_block.params
    }
    model = model_cls(**model_kwargs)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline
