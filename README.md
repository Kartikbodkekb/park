---
title: Park Environment Server
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Park Environment

An urban parking RL environment simulating congested Indian city parking scenarios. An AI agent must navigate real-world factors — traffic, competition, road blockages, and fuel limits — to find and secure parking at the lowest cost in the shortest time.

Inspired by real parking conditions in Pune, India across three zone types: quiet residential areas, busy shopping districts, and high-congestion festival markets (Tulshibaug / Dagdusheth).

## Project Structure

```
park-main/
├── Dockerfile                  # Container image definition
├── README.md                   # This file
├── openenv.yaml                # OpenEnv manifest with task definitions
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependencies (generated)
├── .gitignore
├── __init__.py                 # Module exports (ParkAction, ParkObservation, ParkEnv)
├── client.py                   # ParkEnv HTTP client
├── models.py                   # Action, Observation, and Reward Pydantic models
├── inference.py                # Baseline inference script (runs all 3 tasks)
└── server/
    ├── __init__.py             # Server module exports
    ├── park_environment.py     # Core environment logic (step, reset, grade)
    ├── app.py                  # FastAPI application (HTTP + WebSocket endpoints)
    └── requirements.txt        # Server dependencies
```

## Setup & Running

### Prerequisites

- Python 3.10+
- Docker
- `uv` (recommended) or `pip`
- OpenEnv: `pip install openenv-core`

### Option 1 — Run Locally (without Docker)

```bash
# Clone the repo
git clone https://github.com/Kartikbodkekb/park.git
cd park-main

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`. Visit `http://localhost:8000/web` for the interactive web UI and `http://localhost:8000/docs` for the API docs.

### Option 2 — Run with Docker

```bash
# From project root — build the Docker image
docker build -t park-env:latest .

# Run the container
docker run -p 8000:8000 park-env:latest
```

### Running the Baseline Inference Script

Set up your `.env` file (or export variables directly):

```bash
HF_TOKEN=your_huggingface_token
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

Then run:

```bash
python inference.py
```

Expected output format:

```
============================================================
Running task: EASY
============================================================
[START] task=easy env=smart_parking_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=move_to_nearby reward=0.83 done=true error=null
[END] success=true steps=1 score=0.890 rewards=0.83

============================================================
Running task: MEDIUM
============================================================
[START] task=medium env=smart_parking_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=move_to_nearby reward=0.45 done=true error=null
[END] success=true steps=1 score=0.846 rewards=0.45

============================================================
Running task: HARD
============================================================
[START] task=hard env=smart_parking_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=wait reward=-0.27 done=false error=Agent waited one step. A slot opened up nearby!
[STEP] step=2 action=move_to_nearby reward=-0.11 done=true error=null
[END] success=true steps=2 score=0.776 rewards=-0.27,-0.11

============================================================
FINAL SCORES
============================================================
  easy      : 0.8900 (Paid: Rs.14)
  medium    : 0.8460 (Paid: Rs.42)
  hard      : 0.7765 (Paid: Rs.98)
  average   : 0.8375
============================================================
```

## Quick Start (Client Usage)

Once the server is running, interact with it using the `ParkEnv` client:

```python
from client import ParkEnv
from models import ParkAction

try:
    parkenv = ParkEnv(base_url="http://localhost:8000", task="easy")

    # Reset environment
    obs = parkenv.reset()
    print(f"Zone: {obs.zone_type}")
    print(f"Nearby slots: {obs.nearby_slots}")
    print(f"Traffic: {obs.traffic_level}")

    # Run a few steps
    actions = ["move_to_nearby", "wait", "move_to_far"]
    for act in actions:
        obs, reward, done, info = parkenv.step(ParkAction(action=act))
        print(f"Action: {act} → Reward: {reward:.2f}, Parked: {obs.parked}")
        if done:
            break

finally:
    parkenv.close()
```

## Environment Details

### Tasks

Three difficulty levels, each simulating a different real-world parking scenario:

| Task | Zone | Description | Max Steps |
|------|------|-------------|-----------|
| `easy` | Residential | Quiet area, low traffic, ample slots, government parking nearby | 20 |
| `medium` | Shopping | Busy mall area, moderate competition, limited private slots | 30 |
| `hard` | Market (Festival) | Tulshibaug/Dagdusheth festival zone, very high congestion, scarce and expensive parking | 40 |

### Action Space

**ParkAction**: One of five discrete actions

| Action | Description |
|--------|-------------|
| `move_to_nearby` | Move toward the nearby parking area |
| `move_to_far` | Move toward the distant (cheaper/emptier) parking area |
| `explore_random` | Search randomly in surrounding streets for a free slot |
| `wait` | Stay in place and wait for a slot to open up |
| `leave_area` | Give up and leave without parking (episode ends) |

### Observation Space

**ParkObservation**: Full state of the parking environment at each step

**Zone context:**
- `zone_type` (str) — Type of urban zone: `residential` / `shopping` / `market`
- `time_of_day` (str) — `morning` / `afternoon` / `evening` / `night`
- `day_type` (str) — `weekday` / `weekend`
- `is_festival` (bool) — True if a local festival is active (significantly increases congestion and competition)

**Traffic & competition:**
- `traffic_level` (str) — `low` / `medium` / `high` / `very_high`
- `competition_level` (str) — `low` / `medium` / `high` (number of competing drivers)

**Nearby parking option (closer, usually more expensive):**
- `nearby_slots` (int) — Estimated available slots in the nearby parking area
- `nearby_distance` (float) — Walking distance to nearby parking in km
- `nearby_price` (float) — Hourly parking price at nearby area in INR
- `nearby_type` (str) — `private` / `government` / `street`
- `nearby_facilities` (list[str]) — Available facilities e.g. `covered`, `ev_charging`

**Far parking option (farther, usually cheaper):**
- `far_slots` (int) — Estimated available slots in the far parking area
- `far_distance` (float) — Walking distance to far parking in km
- `far_price` (float) — Hourly parking price at far area in INR
- `far_type` (str) — `private` / `government` / `street`
- `far_facilities` (list[str]) — Available facilities e.g. `open`, `covered`

**Dynamic events:**
- `crowd_spike` (bool) — Sudden crowd surge active (increases competition and congestion)
- `road_blocked` (bool) — A road blockage is active (movement actions may fail)

**Agent status:**
- `fuel_level` (float) — Remaining fuel as a fraction from 0.0 (empty) to 1.0 (full)
- `time_elapsed` (int) — Number of steps taken so far in this episode
- `last_action_result` (str | None) — Human-readable result of the last action taken
- `parked` (bool) — True if the agent has successfully parked
- `price_paid` (float) — Price paid for parking in INR (0.0 if not parked yet)

### Reward Function

The reward is a dense multi-component signal provided at every step — not just at the end of the episode.

| Component | Value | When Triggered |
|-----------|-------|----------------|
| `success_bonus` | +1.0 | Agent successfully parks |
| `movement_reward` | +0.05 to +0.1 | Moving toward a parking area with available slots |
| `movement_reward` | -0.1 | Moving toward a parking area with no slots |
| `wait_penalty` | -0.05 | Agent chooses to wait |
| `traffic_penalty` | -0.1 | High or very_high traffic level |
| `traffic_penalty` | additional -0.1 | Crowd spike is active |
| `blockage_penalty` | -0.2 | Agent hits a blocked road |
| `fuel_penalty` | -0.02 | Per step fuel consumption (movement costs more) |
| `time_penalty` | -0.01 | Per step time cost |
| `cost_penalty` | proportional | Penalty based on parking price paid (price × 0.01) |

### Grader

After each episode, `grade()` returns a score between **0.0 and 1.0**:

```
score = 0.4 × success + 0.2 × time_score + 0.2 × fuel_score + 0.2 × cost_score
```

| Component | Weight | Description |
|-----------|--------|-------------|
| `success` | 40% | 1.0 if parked, 0.0 if not |
| `time_score` | 20% | How quickly the agent parked (steps remaining / max steps) |
| `fuel_score` | 20% | How much fuel remained at end of episode |
| `cost_score` | 20% | How cheap the parking was (lower price = higher score) |

An agent that parks quickly, cheaply, and with fuel to spare scores close to 1.0. An agent that fails to park scores at most 0.6 (from time + fuel + cost components even without success).

### Baseline Scores

Approximate scores using `Qwen/Qwen2.5-72B-Instruct` as the baseline agent:

| Task | Typical Score Range |
|------|-------------------|
| `easy` | 0.75 – 0.90 |
| `medium` | 0.70 – 0.85 |
| `hard` | 0.50 – 0.78 |

## API Reference

### POST `/reset`

Reset the environment and start a new episode.

```json
{ "task": "easy" }
```

Returns a `State` object with the initial observation.

### POST `/step`

Execute one action in the environment.

```json
{
  "action": {
    "action": "move_to_nearby",
    "task": "easy"
  }
}
```

Returns a `State` object with the new observation, reward, done flag, and info.

### GET `/state`

Get the current environment state without taking an action. Returns current episode metadata including `episode_id`, `step_count`, `done`, `task`, and `observation`.

### GET `/grade`

Returns the final score (0.0 → 1.0) for the completed episode.

```json
{ "score": 0.823 }
```

### GET `/health`

Health check endpoint. Returns `{"status": "ok"}` when the server is running.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes (for inference) | Your Hugging Face API token |
| `API_BASE_URL` | No | LLM API base URL (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | No | Model to use for inference (default: `Qwen/Qwen2.5-72B-Instruct`) |
| `LOCAL_IMAGE_NAME` | No | Docker image name for local runs (default: `park-env:latest`) |

## Testing

```bash
# Test environment logic directly (no server required)
python3 server/park_environment.py

# Run test suite
python3 -m pytest tests/ -v
```