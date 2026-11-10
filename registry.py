# registry.py
from typing import Any, Dict, List, Optional, Tuple

class Dataset:
    def __init__(self, name: str, version: str, data_shape: Tuple[int, ...], metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.version = version
        self.data_shape = data_shape
        self.metadata = metadata or {}

    def validate(self) -> bool:
        if not self.name or not self.version:
            raise ValueError("Dataset must have a valid name and version string.")
        if not self.data_shape or any(dim <= 0 for dim in self.data_shape):
            raise ValueError("Dataset data_shape dimensions must be positive integers.")
        return True

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"

class Model:
    def __init__(self, name: str, version: str, framework: str, hyperparameters: Optional[Dict[str, Any]] = None, metrics: Optional[Dict[str, float]] = None):
        self.name = name
        self.version = version
        self.framework = framework
        self.hyperparameters = hyperparameters or {}
        self.metrics = metrics or {}
        self.dataset_id: Optional[str] = None

    def link_dataset(self, dataset: Dataset) -> None:
        dataset.validate()
        self.dataset_id = dataset.identifier

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"

class ModelRegistry:
    def __init__(self):
        self._registry: Dict[str, Model] = {}

    def register(self, model: Model) -> str:
        key = model.identifier
        if key in self._registry:
            raise ValueError(f"Model artifact '{key}' is already registered.")
        self._registry[key] = model
        return key

    def get(self, name: str, version: str) -> Model:
        key = f"{name}:{version}"
        if key not in self._registry:
            raise KeyError(f"Model artifact '{key}' not found in registry.")
        return self._registry[key]

    def list_models(self) -> List[Model]:
        return list(self._registry.values())

    def search_by_framework(self, framework: str) -> List[Model]:
        return [m for m in self._registry.values() if m.framework.lower() == framework.lower()]