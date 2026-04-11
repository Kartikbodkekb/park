# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Park Environment.

The park environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel
from typing import List, Literal, Optional

# ****************************** Action *********************************
class ParkAction(Action):
    """
    The action the agent takes at each step.
    One of five possible actions.
    """
    
    # ✅ ADD THIS: Task selector
    task: Literal["easy", "medium", "hard"] = Field(
        default="easy",
        description="Select the difficulty level"
    )
    
    action: Literal[
        "move_to_nearby",   # Move towards the nearby parking area
        "move_to_far",      # Move towards a distant (cheaper/emptier) parking area
        "explore_random",   # Search randomly in surrounding streets
        "wait",             # Stay and wait for a slot to open up
        "leave_area",       # Give up and leave without parking
    ] = Field(
        ...,
        description="The action to take in the current step."
    )

# ****************************** Obseravtion *********************************
class ParkObservation(Observation):
    zone_type: Literal["residential", "shopping", "market"] = Field(
        description="Type of urban zone being navigated."
    )
    time_of_day: Literal["morning", "afternoon", "evening", "night"] = Field(
        description="Current time of day."
    )
    day_type: Literal["weekday", "weekend"] = Field(
        description="Whether it's a weekday or weekend."
    )
    is_festival: bool = Field(
        description="True if a local festival is active (increases congestion)."
    )

    # traffic and competition
    traffic_level: Literal["low", "medium", "high", "very_high"] = Field(
        description="Current road traffic intensity."
    )
    competition_level: Literal["low", "medium", "high"] = Field(
        description="Number of competing drivers looking for parking."
    )

    # nearby options
    nearby_slots: int = Field(
        ge=0, description="Estimated available slots in the nearby parking area."
    )
    nearby_distance: float = Field(
        ge=0.0, description="Walking distance to nearby parking (in km)."
    )
    nearby_price: float = Field(
        ge=0.0, description="Hourly parking price at nearby area (in INR)."
    )
    nearby_type: Literal["private", "government", "street"] = Field(
        description="Type of nearby parking."
    )
    nearby_facilities: List[str] = Field(
        default_factory=list,
        description="Facilities at nearby parking (e.g. covered, ev_charging)."
    )

    # far options
    far_slots: int = Field(
        ge=0, description="Estimated available slots in the far parking area."
    )
    far_distance: float = Field(
        ge=0.0, description="Walking distance to far parking (in km)."
    )
    far_price: float = Field(
        ge=0.0, description="Hourly parking price at far area (in INR)."
    )
    far_type: Literal["private", "government", "street"] = Field(
        description="Type of far parking."
    )
    far_facilities: List[str] = Field(
        default_factory=list,
        description="Facilities at far parking (e.g. open, covered)."
    )

    # crowd spikes and blockages
    crowd_spike: bool = Field(
        description="Sudden crowd surge active — increases competition and congestion."
    )
    road_blocked: bool = Field(
        description="A road blockage is active — movement actions may fail."
    )

    # status
    fuel_level: float = Field(
        ge=0.0, le=1.0,
        description="Remaining fuel as a fraction (0.0 = empty, 1.0 = full)."
    )
    time_elapsed: int = Field(
        ge=0, description="Number of steps elapsed in this episode."
    )

    last_action_result: Optional[str] = Field(
        default=None,
        description="Human-readable result of the last action taken."
    )
    parked: bool = Field(
        default=False,
        description="True if the agent has successfully parked."
    )
    price_paid: float = Field(
        default=0.0,
        description="Price paid for parking (0 if not parked yet)."
    )


# ****************************** Reward *********************************
class ParkReward(BaseModel):
    total_reward: float = Field(
        description="Scalar reward signal for this step."
    )
    success_bonus: float = Field(
        default=0.0, description="Bonus for successfully parking (+1.0)."
    )
    movement_reward: float = Field(
        default=0.0, description="Reward for moving towards a good option."
    )
    wait_penalty: float = Field(
        default=0.0, description="Penalty for waiting (-0.05)."
    )
    traffic_penalty: float = Field(
        default=0.0, description="Penalty for high traffic conditions."
    )
    blockage_penalty: float = Field(
        default=0.0, description="Penalty for hitting a blocked road."
    )
    fuel_penalty: float = Field(
        default=0.0, description="Penalty for fuel consumed this step."
    )
    time_penalty: float = Field(
        default=0.0, description="Per-step time cost."
    )
    cost_penalty: float = Field(
        default=0.0, description="Penalty proportional to parking price paid."
    )
