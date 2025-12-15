# SPDX-License-Identifier: Apache-2.0
"""
Gateway-based service discovery for seed instance auto-discovery.

This module provides functionality to:
1. Discover seed instances through the Gateway API
2. Register the current instance as a seed instance
3. Query available seed instances for weight loading
"""

import json
import logging
import socket
from dataclasses import dataclass
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# Label keys for seed instance metadata
SEED_INSTANCE_LABEL = "sglang.ai/seed-instance"
SEED_SERVICE_PORT_LABEL = "sglang.ai/seed-service-port"
SEED_GROUP_PORTS_LABEL = "sglang.ai/seed-group-ports"
SEED_MODEL_PATH_LABEL = "sglang.ai/model-path"
SEED_TP_SIZE_LABEL = "sglang.ai/tp-size"


@dataclass
class SeedInstanceInfo:
    """Information about a seed instance that can provide model weights."""

    ip: str
    service_port: int
    group_ports: List[int]
    model_path: Optional[str] = None
    tp_size: Optional[int] = None
    url: Optional[str] = None
    is_healthy: bool = True


def get_local_ip() -> str:
    """Get the local IP address of this machine."""
    return socket.gethostbyname(socket.gethostname())


def register_as_seed_instance(
    gateway_url: str,
    instance_ip: str,
    service_port: int,
    group_ports: List[int],
    model_path: str,
    tp_size: int,
    timeout: float = 30.0,
) -> bool:
    """
    Register the current instance as a seed instance with the Gateway.

    Args:
        gateway_url: URL of the Gateway (e.g., "http://gateway:30000")
        instance_ip: IP address of this instance
        service_port: HTTP service port of this instance
        group_ports: List of NCCL communication group ports for weight transfer
        model_path: Model path/name being served
        tp_size: Tensor parallelism size
        timeout: Request timeout in seconds

    Returns:
        True if registration was successful, False otherwise
    """
    worker_url = f"http://{instance_ip}:{service_port}"

    try:
        response = requests.post(
            f"{gateway_url.rstrip('/')}/workers",
            json={
                "url": worker_url,
                "model_id": model_path,
                "labels": {
                    SEED_INSTANCE_LABEL: "true",
                    SEED_SERVICE_PORT_LABEL: str(service_port),
                    SEED_GROUP_PORTS_LABEL: json.dumps(group_ports),
                    SEED_MODEL_PATH_LABEL: model_path,
                    SEED_TP_SIZE_LABEL: str(tp_size),
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        logger.info(
            f"Successfully registered as seed instance at {worker_url} "
            f"with Gateway {gateway_url}"
        )
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to register as seed instance with Gateway: {e}")
        return False


def unregister_seed_instance(
    gateway_url: str,
    instance_ip: str,
    service_port: int,
    timeout: float = 10.0,
) -> bool:
    """
    Unregister this instance as a seed instance from the Gateway.

    Args:
        gateway_url: URL of the Gateway
        instance_ip: IP address of this instance
        service_port: HTTP service port of this instance
        timeout: Request timeout in seconds

    Returns:
        True if unregistration was successful, False otherwise
    """
    worker_url = f"http://{instance_ip}:{service_port}"

    try:
        # URL encode the worker URL for the path parameter
        import urllib.parse

        encoded_url = urllib.parse.quote(worker_url, safe="")
        response = requests.delete(
            f"{gateway_url.rstrip('/')}/workers/{encoded_url}",
            timeout=timeout,
        )
        response.raise_for_status()
        logger.info(f"Successfully unregistered seed instance {worker_url}")
        return True
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to unregister seed instance: {e}")
        return False


def discover_seed_instances(
    gateway_url: str,
    model_path: Optional[str] = None,
    tp_size: Optional[int] = None,
    timeout: float = 30.0,
) -> List[SeedInstanceInfo]:
    """
    Discover available seed instances through the Gateway.

    Args:
        gateway_url: URL of the Gateway (e.g., "http://gateway:30000")
        model_path: Optional model path to filter by
        tp_size: Optional tensor parallelism size to filter by
        timeout: Request timeout in seconds

    Returns:
        List of SeedInstanceInfo for available seed instances
    """
    try:
        # First try the dedicated seed-instances endpoint
        params = {}
        if model_path:
            params["model_path"] = model_path
        if tp_size:
            params["tp_size"] = str(tp_size)

        try:
            response = requests.get(
                f"{gateway_url.rstrip('/')}/seed-instances",
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return [
                SeedInstanceInfo(**info) for info in data.get("seed_instances", [])
            ]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Endpoint not available, fall back to /workers
                logger.debug(
                    "Gateway /seed-instances endpoint not available, "
                    "falling back to /workers"
                )
            else:
                raise

        # Fall back to querying /workers and filtering
        response = requests.get(
            f"{gateway_url.rstrip('/')}/workers",
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        seed_instances = []
        for worker in data.get("workers", []):
            metadata = worker.get("metadata", {})

            # Check if this is a seed instance
            if metadata.get(SEED_INSTANCE_LABEL) != "true":
                continue

            # Check health
            if not worker.get("is_healthy", False):
                continue

            # Filter by model_path if specified
            worker_model_path = metadata.get(SEED_MODEL_PATH_LABEL)
            if model_path and worker_model_path != model_path:
                continue

            # Filter by tp_size if specified
            worker_tp_size = metadata.get(SEED_TP_SIZE_LABEL)
            if tp_size and worker_tp_size and int(worker_tp_size) != tp_size:
                continue

            # Parse the group ports
            try:
                group_ports = json.loads(
                    metadata.get(SEED_GROUP_PORTS_LABEL, "[]")
                )
            except json.JSONDecodeError:
                logger.warning(
                    f"Invalid group ports format for worker {worker.get('url')}"
                )
                continue

            # Extract IP from worker URL
            worker_url = worker.get("url", "")
            try:
                from urllib.parse import urlparse

                parsed = urlparse(worker_url)
                ip = parsed.hostname
                if not ip:
                    continue
            except Exception:
                continue

            seed_instances.append(
                SeedInstanceInfo(
                    ip=ip,
                    service_port=int(
                        metadata.get(SEED_SERVICE_PORT_LABEL, parsed.port or 8000)
                    ),
                    group_ports=group_ports,
                    model_path=worker_model_path,
                    tp_size=int(worker_tp_size) if worker_tp_size else None,
                    url=worker_url,
                    is_healthy=worker.get("is_healthy", True),
                )
            )

        return seed_instances

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to discover seed instances from Gateway: {e}")
        return []


def select_best_seed_instance(
    seed_instances: List[SeedInstanceInfo],
    preferred_model_path: Optional[str] = None,
    required_tp_size: Optional[int] = None,
) -> Optional[SeedInstanceInfo]:
    """
    Select the best seed instance from a list of candidates.

    Selection criteria:
    1. Must match required_tp_size if specified
    2. Prefer instances with matching model_path
    3. Prefer healthy instances

    Args:
        seed_instances: List of candidate seed instances
        preferred_model_path: Preferred model path to match
        required_tp_size: Required tensor parallelism size

    Returns:
        The best matching SeedInstanceInfo, or None if no suitable instance found
    """
    if not seed_instances:
        return None

    candidates = seed_instances.copy()

    # Filter by required_tp_size
    if required_tp_size is not None:
        candidates = [
            s for s in candidates if s.tp_size is None or s.tp_size == required_tp_size
        ]
        if not candidates:
            logger.warning(
                f"No seed instances found with tp_size={required_tp_size}"
            )
            return None

    # Filter by health
    healthy_candidates = [s for s in candidates if s.is_healthy]
    if healthy_candidates:
        candidates = healthy_candidates

    # Prefer matching model_path
    if preferred_model_path:
        matching = [s for s in candidates if s.model_path == preferred_model_path]
        if matching:
            candidates = matching

    # Return the first candidate
    return candidates[0] if candidates else None


def wait_for_seed_instance(
    gateway_url: str,
    model_path: Optional[str] = None,
    tp_size: Optional[int] = None,
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> Optional[SeedInstanceInfo]:
    """
    Wait for a seed instance to become available.

    Args:
        gateway_url: URL of the Gateway
        model_path: Optional model path to filter by
        tp_size: Optional tensor parallelism size to filter by
        timeout: Maximum time to wait in seconds
        poll_interval: Time between polls in seconds

    Returns:
        SeedInstanceInfo if found, None if timeout
    """
    import time

    start_time = time.time()
    attempt = 0

    while time.time() - start_time < timeout:
        attempt += 1
        if attempt % 10 == 1:  # Log every 10 attempts
            logger.info(
                f"Waiting for seed instance (attempt {attempt}, "
                f"elapsed {time.time() - start_time:.1f}s)..."
            )

        seed_instances = discover_seed_instances(
            gateway_url=gateway_url,
            model_path=model_path,
            tp_size=tp_size,
            timeout=min(poll_interval * 2, 30.0),
        )

        best = select_best_seed_instance(
            seed_instances,
            preferred_model_path=model_path,
            required_tp_size=tp_size,
        )

        if best:
            logger.info(
                f"Found seed instance: {best.ip}:{best.service_port} "
                f"(model: {best.model_path}, tp_size: {best.tp_size})"
            )
            return best

        time.sleep(poll_interval)

    logger.error(
        f"Timeout waiting for seed instance after {timeout}s "
        f"(model_path={model_path}, tp_size={tp_size})"
    )
    return None

