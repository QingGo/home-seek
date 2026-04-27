"""Expert prediction strategies for prefetch.

Each strategy predicts which experts will be routed for a given layer,
based on available context (hidden state, input ids, etc.).

Plug in different strategies without changing inference loop.
"""

from abc import ABC, abstractmethod
import torch


class PredictResult:
    def __init__(self, expert_ids: list[int], confidence: float = 1.0):
        self.expert_ids = expert_ids
        self.confidence = confidence


class ExpertPredictor(ABC):
    @abstractmethod
    def predict(
        self,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        layer_idx: int,
    ) -> PredictResult:
        ...

    def collect(
        self,
        layer_idx: int,
        hidden_state: torch.Tensor,
        actual_topk: torch.Tensor,
    ):
        """Record actual routing outcome for training data collection."""
        pass


class HashExpertPredictor(ExpertPredictor):
    """100% accurate for hash layers: routing determined by token id."""

    def __init__(self, tid2eid: torch.Tensor, topk: int = 6):
        self._tid2eid = tid2eid
        self._topk = topk

    def predict(self, hidden_state, input_ids, layer_idx):
        eids = self._tid2eid[input_ids]
        flat = eids.reshape(-1, eids.shape[-1])
        return PredictResult(flat[0].tolist()[:self._topk], confidence=1.0)

    def collect(self, layer_idx, hidden_state, actual_topk):
        pass


class OraclePredictor(ExpertPredictor):
    """Ground truth: actually runs routing to get exact result.
    Only for profiling upper bound / data collection.
    """

    def __init__(self, topk: int = 6):
        self._topk = topk
        self._history: list[dict] = []

    def predict(self, hidden_state, input_ids, layer_idx):
        return PredictResult([], confidence=0.0)

    def set_routing_result(self, layer_idx: int, expert_ids, weights):
        self._history.append({
            "layer": layer_idx,
            "expert_ids": expert_ids.clone() if torch.is_tensor(expert_ids) else expert_ids,
        })

    def get_last_prediction(self, layer_idx: int):
        for entry in reversed(self._history):
            if entry["layer"] == layer_idx:
                return PredictResult(entry["expert_ids"].reshape(-1).tolist(), confidence=1.0)
        return PredictResult([], confidence=0.0)


class RecordingPredictor(ExpertPredictor):
    """Wraps a predictor and records (hidden, actual) pairs for offline training."""

    def __init__(self, inner: ExpertPredictor, dataset_size: int = 10000):
        self._inner = inner
        self._dataset: list[tuple[torch.Tensor, int, torch.Tensor]] = []
        self._dataset_size = dataset_size

    def predict(self, hidden_state, input_ids, layer_idx):
        return self._inner.predict(hidden_state, input_ids, layer_idx)

    def collect(self, layer_idx, hidden_state, actual_topk):
        if len(self._dataset) < self._dataset_size:
            self._dataset.append((hidden_state.detach().cpu(), layer_idx, actual_topk.detach().cpu()))

    def save_dataset(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._dataset, f)
        print(f"[RecordingPredictor] Saved {len(self._dataset)} samples to {path}")


class HeuristicPredictor(ExpertPredictor):
    """Predict next layer's experts based on current layer routing.

    Heuristics:
    - same-as-current: next layer routes same experts as current layer
    - hash-layer-prior: for hash layers, use token-to-expert mapping
    """

    def __init__(self, num_layers: int, topk: int = 6):
        self._num_layers = num_layers
        self._topk = topk
        self._last_routing: dict[int, list[int]] = {}

    def predict(self, hidden_state, input_ids, layer_idx):
        next_idx = layer_idx + 1
        if next_idx >= self._num_layers:
            return PredictResult([], confidence=0.0)

        # If we have recorded routing for next_idx, use it
        if next_idx in self._last_routing:
            return PredictResult(self._last_routing[next_idx], confidence=0.6)

        # Fall back to current layer's routing (same-as-current heuristic)
        if layer_idx in self._last_routing:
            return PredictResult(self._last_routing[layer_idx], confidence=0.4)

        return PredictResult([], confidence=0.0)

    def collect(self, layer_idx, hidden_state, actual_topk):
        flat = actual_topk.reshape(-1, actual_topk.shape[-1])
        eids = []
        for k in range(flat.shape[1]):
            eid = int(flat[0, k].item())
            if eid >= 0:
                eids.append(eid)
        self._last_routing[layer_idx] = eids
