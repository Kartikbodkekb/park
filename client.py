# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Park Environment Client."""

# from typing import Dict
import os
from typing import Any, Dict, Optional, Tuple

import requests


from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from park.models import ParkAction, ParkObservation


class ParkEnv:
    VALID_TASKS = ["easy", "medium", "hard"]

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        task: str = "easy",
        timeout: int = 30,
    ):
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}")
        self.base_url = base_url.rstrip("/")
        self.task = task
        self.timeout = timeout
        self._session = requests.Session()

    
    def reset(self) -> ParkObservation:
        """Start a new episode. Returns the initial observation."""
        resp = self._post("/reset", {"task": self.task})
        return ParkObservation(**resp["observation"])

    def step(
        self, action: ParkAction
    ) -> Tuple[ParkObservation, float, bool, Dict[str, Any]]:
        """
        Execute one action.
        Returns (observation, reward, done, info).
        """
        payload = {
            "action": action.model_dump(),
            "task": self.task,
        }
        resp = self._post("/step", payload)
        obs = ParkObservation(**resp["observation"])
        return obs, resp["reward"], resp["done"], resp["info"]

    def state(self) -> Dict[str, Any]:
        """Return current episode metadata."""
        return self._get("/state", params={"task": self.task})

    def grade(self) -> float:
        """Return the final score (0.0 → 1.0) for the completed episode."""
        resp = self._get("/grade", params={"task": self.task})
        return resp["score"]

    def health(self) -> bool:
        """Check if the server is up."""
        try:
            resp = self._get("/health")
            return resp.get("status") == "ok"
        except Exception:
            return False

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    
    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self._session.post(url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    