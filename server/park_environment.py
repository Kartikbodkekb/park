# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Park Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4
from typing import Any,Dict,Optional,Tuple,List
import random
import os
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from pydantic import BaseModel

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ParkAction, ParkObservation , ParkReward
except ImportError:
    from models import ParkAction, ParkObservation , ParkReward

# **********************Task Configs*****************************

TASK_CONFIGS = {
    "easy": {
        "zone_type": "residential",
        "description": "Quiet residential area — low traffic, ample parking.",
        "max_time": 20,
        "max_fuel": 1.0,
        "max_price": 30.0,
        "nearby_slots_range": (5, 10),
        "nearby_price_range": (10.0, 20.0),
        "nearby_distance_range": (0.1, 0.3),
        "nearby_type": "government",
        "nearby_facilities": ["covered"],
        "far_slots_range": (8, 15),
        "far_price_range": (5.0, 15.0),
        "far_distance_range": (0.4, 0.8),
        "far_type": "street",
        "far_facilities": ["open"],
        "traffic_weights": {"low": 0.6, "medium": 0.3, "high": 0.1, "very_high": 0.0},
        "competition_weights": {"low": 0.7, "medium": 0.2, "high": 0.1},
        "crowd_spike_prob": 0.05,
        "road_block_prob": 0.03,
        "time_of_day": "morning",
        "day_type": "weekday",
        "is_festival": False,
    },
    "medium": {
        "zone_type": "shopping",
        "description": "Busy shopping area — moderate traffic, limited slots.",
        "max_time": 30,
        "max_fuel": 1.0,
        "max_price": 60.0,
        "nearby_slots_range": (2, 6),
        "nearby_price_range": (30.0, 50.0),
        "nearby_distance_range": (0.1, 0.2),
        "nearby_type": "private",
        "nearby_facilities": ["covered", "ev_charging"],
        "far_slots_range": (4, 8),
        "far_price_range": (15.0, 30.0),
        "far_distance_range": (0.5, 1.2),
        "far_type": "government",
        "far_facilities": ["open"],
        "traffic_weights": {"low": 0.1, "medium": 0.5, "high": 0.3, "very_high": 0.1},
        "competition_weights": {"low": 0.2, "medium": 0.5, "high": 0.3},
        "crowd_spike_prob": 0.15,
        "road_block_prob": 0.10,
        "time_of_day": "afternoon",
        "day_type": "weekend",
        "is_festival": False,
    },
    "hard": {
        "zone_type": "market",
        "description": "Tulshibaug / Dagdusheth Festival — very high congestion, scarce parking.",
        "max_time": 40,
        "max_fuel": 1.0,
        "max_price": 100.0,
        "nearby_slots_range": (0, 2),
        "nearby_price_range": (60.0, 100.0),
        "nearby_distance_range": (0.05, 0.15),
        "nearby_type": "private",
        "nearby_facilities": [],
        "far_slots_range": (1, 4),
        "far_price_range": (30.0, 60.0),
        "far_distance_range": (1.0, 2.5),
        "far_type": "street",
        "far_facilities": ["open"],
        "traffic_weights": {"low": 0.0, "medium": 0.1, "high": 0.4, "very_high": 0.5},
        "competition_weights": {"low": 0.0, "medium": 0.2, "high": 0.8},
        "crowd_spike_prob": 0.40,
        "road_block_prob": 0.25,
        "time_of_day": "evening",
        "day_type": "weekend",
        "is_festival": True,
    },
}


def _weighted_choice(weights: Dict[str, float]) -> str:
    keys = list(weights.keys())
    probs = list(weights.values())
    return random.choices(keys, weights=probs, k=1)[0]


class ParkStateResponse(BaseModel):
    episode_id: str
    step_count: int
    done: bool
    task: str
    observation: Optional[Dict[str, Any]] = None



class ParkEnvironment(Environment):

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    VALID_TASKS = ["easy", "medium", "hard"]

    def __init__(self, task: str = "easy"):
        """Initialize the Park Environment."""
        # Initialize state variables FIRST
        self._episode_id: str = ""
        self._step_count: int = 0
        self._done: bool = False
        self._obs: Optional[ParkObservation] = None
        self.task: str = ""
        self.cfg: Dict = {}
    
        # Then set task
        self.set_task(task)
        
        # Initialize observation
        self._obs = self._build_initial_obs()  # ✅ Initialize _obs before use

    def set_task(self, task: str):
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}, got '{task}'")
        self.task = task
        self.cfg = TASK_CONFIGS[task]

    @classmethod
    def from_docker_image(cls, image_name: str = None):
        """Mandatory for Scaler/Meta PyTorch Hackathon spec."""
        return cls()


    def reset(self, task: Optional[str] = None) -> State:
        """Reset the environment and return initial state."""
        if task:
            self.set_task(task)
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._done = False
        self._obs = self._build_initial_obs()
        
        # ── FIX: Return State object instead of just observation ──
        return State(
            observation=self._obs,
            reward=0.0,
            done=False,
            info={"episode_id": self._episode_id}
        )

    def step(self, action_input: Any) -> ParkObservation:
        """Execute one step and return the observation."""
        if self._done:
            raise RuntimeError("Episode is over. Call reset() to start a new one.")

        act = action_input.action if hasattr(action_input, 'action') else action_input

        self._step_count += 1
        reward_breakdown = ParkReward(total_reward=0.0)
        result_msg = ""

        obs = self._obs
        cfg = self.cfg

        # 1. Environment Dynamic Updates
        obs.traffic_level = _weighted_choice(cfg["traffic_weights"])
        obs.competition_level = _weighted_choice(cfg["competition_weights"])
        obs.crowd_spike = random.random() < cfg["crowd_spike_prob"]
        obs.road_blocked = random.random() < cfg["road_block_prob"]
        obs.nearby_slots = max(0, obs.nearby_slots + random.randint(-2, 1))
        obs.far_slots = max(0, obs.far_slots + random.randint(-1, 1))

        # 2. Fuel and Time Penalties
        fuel_cost = 0.04 if act in ("move_to_nearby", "move_to_far", "explore_random") else 0.01
        obs.fuel_level = max(0.0, obs.fuel_level - fuel_cost)
        reward_breakdown.fuel_penalty = -fuel_cost * 0.5
        reward_breakdown.time_penalty = -0.01
        obs.time_elapsed = self._step_count

        # 3. Action Logic
        if act == "leave_area":
            result_msg = "Agent left the area without parking."
            self._done = True
        elif act == "wait":
            reward_breakdown.wait_penalty = -0.05
            result_msg = "Agent waited one step."
            if random.random() < 0.3:
                obs.nearby_slots = min(obs.nearby_slots + 1, 10)
                result_msg += " A slot opened up nearby!"
        elif act == "explore_random":
            if obs.road_blocked and random.random() < 0.6:
                reward_breakdown.blockage_penalty = -0.2
                result_msg = "Road blocked! Exploration failed."
            else:
                found = random.random() < 0.25
                if found:
                    obs.nearby_slots = max(obs.nearby_slots + 1, 1)
                    reward_breakdown.movement_reward = +0.1
                    result_msg = "Exploration found an extra slot!"
                else:
                    result_msg = "Exploration found nothing new."
        elif act == "move_to_nearby":
            if obs.road_blocked and random.random() < 0.5:
                reward_breakdown.blockage_penalty = -0.2
                result_msg = "Road blocked! Could not reach nearby parking."
            elif obs.nearby_slots > 0:
                success_prob = self._success_probability(obs, "nearby")
                if random.random() < success_prob:
                    obs.parked = True
                    obs.price_paid = obs.nearby_price
                    reward_breakdown.success_bonus = +1.0
                    result_msg = f"Successfully parked nearby! Paid ₹{obs.nearby_price:.0f}/hr."
                    self._done = True
                else:
                    result_msg = "Reached nearby parking but a competitor took the last slot."
                    obs.nearby_slots = max(0, obs.nearby_slots - 1)
                    reward_breakdown.movement_reward = +0.05
            else:
                result_msg = "No slots available at nearby parking."
                reward_breakdown.movement_reward = -0.1
        elif act == "move_to_far":
            if obs.road_blocked and random.random() < 0.3:
                reward_breakdown.blockage_penalty = -0.2
                result_msg = "Road blocked on the way to far parking."
            elif obs.far_slots > 0:
                success_prob = self._success_probability(obs, "far")
                if random.random() < success_prob:
                    obs.parked = True
                    obs.price_paid = obs.far_price
                    reward_breakdown.success_bonus = +1.0
                    result_msg = f"Successfully parked far away. Paid ₹{obs.far_price:.0f}/hr."
                    self._done = True
                else:
                    result_msg = "Reached far parking but a competitor got there first."
                    obs.far_slots = max(0, obs.far_slots - 1)
                    reward_breakdown.movement_reward = +0.05
            else:
                result_msg = "No slots at far parking either."
                reward_breakdown.movement_reward = -0.1

        # 4. Global Penalties
        if obs.traffic_level in ("high", "very_high"):
            reward_breakdown.traffic_penalty = -0.1
        if obs.crowd_spike:
            reward_breakdown.traffic_penalty -= 0.1
        if obs.parked:
            reward_breakdown.cost_penalty = -(obs.price_paid * 0.01)

        # 5. Termination Checks
        if obs.fuel_level <= 0.0 or self._step_count >= cfg["max_time"]:
            self._done = True

        # 6. Final Reward Calculation
        total = sum([
            reward_breakdown.success_bonus, reward_breakdown.movement_reward,
            reward_breakdown.wait_penalty, reward_breakdown.traffic_penalty,
            reward_breakdown.blockage_penalty, reward_breakdown.fuel_penalty,
            reward_breakdown.time_penalty, reward_breakdown.cost_penalty
        ])

        obs.last_action_result = result_msg
        self._obs = obs
        from openenv.core.env_server.types import State

        return State(
            observation=self._obs,
            reward=total,  # This is the 'total' variable you calculated in step 6
            done=self._done,
            info={"episode_id": self._episode_id}
        )
    



    def grade(self) -> float:
        """Grade the current episode."""
        if self._obs is None:
            return 0.0
        cfg, obs = self.cfg, self._obs
        success = 1.0 if obs.parked else 0.0
        time_score = max(0.0, 1.0 - (self._step_count / cfg["max_time"]))
        fuel_score = obs.fuel_level / cfg["max_fuel"]
        cost_score = max(0.0, 1.0 - (obs.price_paid / cfg["max_price"])) if obs.parked else 1.0
        score = (0.4 * success + 0.2 * time_score + 0.2 * fuel_score + 0.2 * cost_score)
        return round(min(1.0, max(0.0, score)), 4)

    @property
    def state(self) -> ParkStateResponse:
        """Return the current state of the environment."""
        return ParkStateResponse(
            episode_id=self._episode_id,
            step_count=self._step_count,
            done=self._done,
            task=self.task,
            observation=self._obs.model_dump() if self._obs else None
        )

    def close(self):
        """Clean up resources."""
        pass

    def _build_initial_obs(self) -> ParkObservation:
        """Build the initial observation for a new episode."""
        cfg = self.cfg
        nearby_slots = random.randint(*cfg["nearby_slots_range"])
        nearby_price = round(random.uniform(*cfg["nearby_price_range"]), 1)
        nearby_dist = round(random.uniform(*cfg["nearby_distance_range"]), 2)
        far_slots = random.randint(*cfg["far_slots_range"])
        far_price = round(random.uniform(*cfg["far_price_range"]), 1)
        far_dist = round(random.uniform(*cfg["far_distance_range"]), 2)

        return ParkObservation(
            zone_type=cfg["zone_type"],
            time_of_day=cfg["time_of_day"],
            day_type=cfg["day_type"],
            is_festival=cfg["is_festival"],
            traffic_level=_weighted_choice(cfg["traffic_weights"]),
            competition_level=_weighted_choice(cfg["competition_weights"]),
            nearby_slots=nearby_slots,
            nearby_distance=nearby_dist,
            nearby_price=nearby_price,
            nearby_type=cfg["nearby_type"],
            nearby_facilities=cfg["nearby_facilities"],
            far_slots=far_slots,
            far_distance=far_dist,
            far_price=far_price,
            far_type=cfg["far_type"],
            far_facilities=cfg["far_facilities"],
            crowd_spike=random.random() < cfg["crowd_spike_prob"],
            road_blocked=random.random() < cfg["road_block_prob"],
            fuel_level=1.0,
            time_elapsed=0,
            last_action_result=None,
            parked=False,
            price_paid=0.0,
        )

    def _success_probability(self, obs: ParkObservation, target: str) -> float:
        """Calculate the probability of successfully parking."""
        comp_penalty = {"low": 0.0, "medium": 0.2, "high": 0.4}[obs.competition_level]
        slots = obs.nearby_slots if target == "nearby" else obs.far_slots
        base = min(1.0, slots / 3.0)
        spike_penalty = 0.2 if obs.crowd_spike else 0.0
        return max(0.05, base - comp_penalty - spike_penalty)