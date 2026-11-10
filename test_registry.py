# test_registry.py
import pytest
from registry import Dataset, Model, ModelRegistry

def test_dataset_validation_success():
    dataset = Dataset("customer_churn", "v1.0", (50000, 25))
    assert dataset.validate() is True
    assert dataset.identifier == "customer_churn:v1.0"

@pytest.mark.parametrize("name, version, shape", [
    ("", "v1.0", (100, 5)),
    ("sales", "", (100, 5)),
    ("sales", "v1.0", ()),
    ("sales", "v1.0", (-10, 5)),
])
def test_dataset_validation_failures(name, version, shape):
    dataset = Dataset(name, version, shape)
    with pytest.raises(ValueError):
        dataset.validate()

def test_model_registration_and_retrieval():
    registry = ModelRegistry()
    model = Model(name="churn_classifier", version="1.0.0", framework="PyTorch", hyperparameters={"lr": 0.001}, metrics={"f1_score": 0.88})
    key = registry.register(model)
    assert key == "churn_classifier:1.0.0"
    retrieved = registry.get("churn_classifier", "1.0.0")
    assert retrieved.framework == "PyTorch"

def test_duplicate_model_registration_raises_error():
    registry = ModelRegistry()
    registry.register(Model("fraud_detector", "v1", "XGBoost"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Model("fraud_detector", "v1", "LightGBM"))

def test_missing_model_raises_key_error():
    registry = ModelRegistry()
    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent_model", "v9.9")

def test_model_dataset_lineage_linking():
    dataset = Dataset("transactions", "v2.1", (100000, 12))
    model = Model("anomaly_detector", "0.1", "Scikit-Learn")
    model.link_dataset(dataset)
    assert model.dataset_id == "transactions:v2.1"

def test_search_models_by_framework():
    registry = ModelRegistry()
    registry.register(Model("model_a", "1.0", "PyTorch"))
    registry.register(Model("model_b", "1.0", "TensorFlow"))
    registry.register(Model("model_c", "2.0", "PyTorch"))
    pytorch_models = registry.search_by_framework("pytorch")
    assert len(pytorch_models) == 2