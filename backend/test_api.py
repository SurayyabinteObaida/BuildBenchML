"""
API-level tests for main.py using FastAPI's TestClient -- exercises the actual
HTTP routes, request/response schemas, and error status codes.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("=== GET /api/health ===")
r = client.get("/api/health")
assert r.status_code == 200
print(r.json())

print("\n=== GET /api/blocks ===")
r = client.get("/api/blocks")
assert r.status_code == 200
blocks = r.json()["blocks"]
print(f"{len(blocks)} blocks returned")
assert len(blocks) == 21
data_blocks = [b for b in blocks if b["category"] == "data"]
assert all("taskType" in b for b in data_blocks)
print("Sample data block:", data_blocks[0])

print("\n=== GET /api/datasets/data:titanic/preview ===")
r = client.get("/api/datasets/data:titanic/preview")
assert r.status_code == 200
preview = r.json()
print("taskType:", preview["taskType"], "| rowCount:", preview["rowCount"],
      "| missing:", preview["missingValueCount"])
assert preview["taskType"] == "classification"
assert preview["missingValueCount"] > 0

print("\n=== GET /api/datasets/nonsense/preview (404 case) ===")
r = client.get("/api/datasets/nonsense/preview")
assert r.status_code == 404
print("Correctly 404s:", r.json())

print("\n=== POST /api/pipeline/run -- valid Iris pipeline ===")
r = client.post("/api/pipeline/run", json={
    "nodes": [
        {"nodeId": "n1", "blockId": "data:iris", "params": {}},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {"test_size": 0.2, "random_state": 42}},
        {"nodeId": "n3", "blockId": "model:logistic_regression", "params": {"C": 1.0, "max_iter": 200}},
        {"nodeId": "n4", "blockId": "eval:accuracy", "params": {}},
    ],
    "edges": [
        {"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n3", "to": "n4"},
    ],
})
assert r.status_code == 200
body = r.json()
print(body)
assert body["success"] is True
assert "accuracy" in body["metrics"]

print("\n=== POST /api/pipeline/run -- Titanic with confusion matrix ===")
r = client.post("/api/pipeline/run", json={
    "nodes": [
        {"nodeId": "n1", "blockId": "data:titanic", "params": {}},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {}},
        {"nodeId": "n3", "blockId": "prep:impute_missing", "params": {"strategy": "median"}},
        {"nodeId": "n4", "blockId": "prep:onehot_encode", "params": {}},
        {"nodeId": "n5", "blockId": "model:random_forest", "params": {"n_estimators": 50, "max_depth": 8}},
        {"nodeId": "n6", "blockId": "eval:accuracy", "params": {}},
        {"nodeId": "n7", "blockId": "eval:confusion_matrix", "params": {}},
    ],
    "edges": [
        {"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n3", "to": "n4"},
        {"from": "n4", "to": "n5"}, {"from": "n5", "to": "n6"}, {"from": "n6", "to": "n7"},
    ],
})
assert r.status_code == 200
body = r.json()
print("accuracy:", body["metrics"]["accuracy"])
print("confusion matrix:", body["metrics"]["confusionMatrix"])
print("labels:", body["metrics"]["confusionMatrixLabels"])
assert "confusionMatrix" in body["metrics"]

print("\n=== POST /api/pipeline/run -- Housing regression ===")
r = client.post("/api/pipeline/run", json={
    "nodes": [
        {"nodeId": "n1", "blockId": "data:housing", "params": {}},
        {"nodeId": "n2", "blockId": "split:train_test", "params": {}},
        {"nodeId": "n3", "blockId": "prep:impute_missing", "params": {"strategy": "median"}},
        {"nodeId": "n4", "blockId": "prep:onehot_encode", "params": {}},
        {"nodeId": "n5", "blockId": "model:linear_regression", "params": {}},
        {"nodeId": "n6", "blockId": "eval:rmse", "params": {}},
        {"nodeId": "n7", "blockId": "eval:r2", "params": {}},
    ],
    "edges": [
        {"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}, {"from": "n3", "to": "n4"},
        {"from": "n4", "to": "n5"}, {"from": "n5", "to": "n6"}, {"from": "n6", "to": "n7"},
    ],
})
assert r.status_code == 200
body = r.json()
print("rmse:", body["metrics"]["rmse"], "| r2:", body["metrics"]["r2"])
assert "rmse" in body["metrics"] and "r2" in body["metrics"]

print("\n=== POST /api/pipeline/run -- invalid pipeline returns 422 with clear message ===")
r = client.post("/api/pipeline/run", json={
    "nodes": [
        {"nodeId": "n1", "blockId": "data:iris", "params": {}},
        {"nodeId": "n2", "blockId": "model:logistic_regression", "params": {}},
        {"nodeId": "n3", "blockId": "eval:accuracy", "params": {}},
    ],
    "edges": [{"from": "n1", "to": "n2"}, {"from": "n2", "to": "n3"}],
})
assert r.status_code == 422
print("Correctly 422s:", r.json())
assert "Train/Test Split" in r.json()["detail"]

print("\n=== POST /api/pipeline/run -- empty request ===")
r = client.post("/api/pipeline/run", json={"nodes": [], "edges": []})
assert r.status_code == 422
print("Correctly 422s:", r.json())

print("\n\nALL API TESTS PASSED")
