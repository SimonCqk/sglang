# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Layer readiness tracker for layer-wise broadcast model loading."""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class LayerReadinessTracker:
    """Thread-safe tracker for layer loading status during layerwise broadcast.

    This class tracks the readiness of model layers during layer-by-layer
    weight transfer, enabling computation overlap where inference can start
    on early layers while later layers are still being transferred.

    Attributes:
        num_layers: Total number of transformer layers in the model.
        layer_ready: List of booleans indicating whether each layer is loaded.
        layer_events: Threading events for blocking until layers are ready.
        embed_ready: Event indicating embedding layer is loaded.
        lm_head_ready: Event indicating lm_head/norm layers are loaded.
    """

    def __init__(self, num_layers: int):
        """Initialize the layer readiness tracker.

        Args:
            num_layers: Total number of transformer layers in the model.
        """
        self.num_layers = num_layers
        self.layer_ready = [False] * num_layers
        self.layer_events = [threading.Event() for _ in range(num_layers)]
        self.embed_ready = threading.Event()
        self.lm_head_ready = threading.Event()
        self._lock = threading.Lock()
        self._all_ready = threading.Event()

    def mark_embed_ready(self) -> None:
        """Mark the embedding layer as ready."""
        self.embed_ready.set()
        logger.debug("Embedding layer marked as ready")

    def mark_layer_ready(self, layer_idx: int) -> None:
        """Mark a specific transformer layer as ready.

        Args:
            layer_idx: Index of the layer to mark as ready.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {self.num_layers})"
            )

        with self._lock:
            self.layer_ready[layer_idx] = True
            self.layer_events[layer_idx].set()

        logger.debug(f"Layer {layer_idx} marked as ready")

        # Check if all layers are now ready
        if self.all_layers_ready():
            self._all_ready.set()

    def mark_lm_head_ready(self) -> None:
        """Mark the lm_head and norm layers as ready."""
        self.lm_head_ready.set()
        logger.debug("LM head and norm layers marked as ready")

    def wait_for_embed(self, timeout: Optional[float] = None) -> bool:
        """Block until the embedding layer is ready.

        Args:
            timeout: Maximum time to wait in seconds. None for no timeout.

        Returns:
            True if the embedding is ready, False if timeout occurred.
        """
        return self.embed_ready.wait(timeout=timeout)

    def wait_for_layer(self, layer_idx: int, timeout: Optional[float] = None) -> bool:
        """Block until a specific layer is ready.

        Args:
            layer_idx: Index of the layer to wait for.
            timeout: Maximum time to wait in seconds. None for no timeout.

        Returns:
            True if the layer is ready, False if timeout occurred.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {self.num_layers})"
            )

        return self.layer_events[layer_idx].wait(timeout=timeout)

    def wait_for_lm_head(self, timeout: Optional[float] = None) -> bool:
        """Block until the lm_head and norm layers are ready.

        Args:
            timeout: Maximum time to wait in seconds. None for no timeout.

        Returns:
            True if lm_head is ready, False if timeout occurred.
        """
        return self.lm_head_ready.wait(timeout=timeout)

    def is_layer_ready(self, layer_idx: int) -> bool:
        """Check if a specific layer is ready without blocking.

        Args:
            layer_idx: Index of the layer to check.

        Returns:
            True if the layer is ready, False otherwise.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {self.num_layers})"
            )

        with self._lock:
            return self.layer_ready[layer_idx]

    def get_max_ready_layer(self) -> int:
        """Get the highest contiguously loaded layer index.

        Returns:
            The highest layer index where all layers from 0 to that index
            are ready. Returns -1 if no layers are ready.
        """
        with self._lock:
            for i in range(self.num_layers):
                if not self.layer_ready[i]:
                    return i - 1
            return self.num_layers - 1

    def get_ready_layer_count(self) -> int:
        """Get the count of layers that are ready.

        Returns:
            Number of layers that have been marked as ready.
        """
        with self._lock:
            return sum(self.layer_ready)

    def all_layers_ready(self) -> bool:
        """Check if all transformer layers are ready.

        Returns:
            True if all layers are ready, False otherwise.
        """
        with self._lock:
            return all(self.layer_ready)

    def is_fully_loaded(self) -> bool:
        """Check if the model is fully loaded (embed + all layers + lm_head).

        Returns:
            True if the entire model is loaded, False otherwise.
        """
        return (
            self.embed_ready.is_set()
            and self.all_layers_ready()
            and self.lm_head_ready.is_set()
        )

    def wait_until_fully_loaded(self, timeout: Optional[float] = None) -> bool:
        """Block until the entire model is loaded.

        Args:
            timeout: Maximum time to wait in seconds. None for no timeout.

        Returns:
            True if fully loaded, False if timeout occurred.
        """
        # Wait for embed first
        if not self.embed_ready.wait(timeout=timeout):
            return False

        # Wait for all layers
        for i in range(self.num_layers):
            if not self.layer_events[i].wait(timeout=timeout):
                return False

        # Wait for lm_head
        return self.lm_head_ready.wait(timeout=timeout)

    def reset(self) -> None:
        """Reset all readiness states to not ready."""
        with self._lock:
            self.layer_ready = [False] * self.num_layers
            for event in self.layer_events:
                event.clear()
            self.embed_ready.clear()
            self.lm_head_ready.clear()
            self._all_ready.clear()

    def __repr__(self) -> str:
        ready_count = self.get_ready_layer_count()
        return (
            f"LayerReadinessTracker("
            f"num_layers={self.num_layers}, "
            f"ready_count={ready_count}, "
            f"embed_ready={self.embed_ready.is_set()}, "
            f"lm_head_ready={self.lm_head_ready.is_set()}"
            f")"
        )
