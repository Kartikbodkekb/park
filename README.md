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

## Quick Start

The simplest way to use the Park environment is through the `ParkEnv` class:

```python
from park import ParkAction, ParkEnv

try:
    # Connect to running HF Space or local server
    parkenv = ParkEnv(base_url="http://localhost:8000", task="easy")

    # Reset environment
    result = parkenv.reset()
    print(f"Zone: {result.zone_type}")
    print(f"Nearby slots: {result.nearby_slots}")
    print(f"Traffic: {result.traffic_level}")

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

The reward is a dense multi-component signal provided at every step — not just at the end of the episode. This gives the agent useful learning signal throughout the episode.

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

After each episode, `grade()` returns a score between **0.0 and 1.0** based on four weighted components:

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
| `easy` | 0.55 – 0.75 |
| `medium` | 0.35 – 0.55 |
| `hard` | 0.10 – 0.30 |

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker
- Hugging Face CLI: `pip install huggingface_hub`
- OpenEnv: `pip install openenv-core`

### Running Locally

```bash
# Clone the repo
git clone https://github.com/Kartikbodkekb/park.git
cd park

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`. Visit `http://localhost:8000/web` for the interactive web UI.

### Building and Running with Docker

```bash
# From project root — build the Docker image
docker build -t park-env:latest .

# Run the container
docker run -p 8000:8000 park-env:latest
```

### Running the Baseline Inference Script

```bash
# Set required environment variables
export HF_TOKEN=your_huggingface_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Run inference across all 3 tasks
python inference.py
```

Expected output format:
```
============================================================
Running task: EASY
============================================================
[START] task=easy env=smart_parking_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=move_to_nearby reward=0.85 done=false error=null
[STEP] step=2 action=move_to_nearby reward=0.74 done=true error=null
[END] success=true steps=2 score=0.823 rewards=0.85,0.74

============================================================
Running task: MEDIUM
============================================================
[START] task=medium env=smart_parking_env model=Qwen/Qwen2.5-72B-Instruct
...

============================================================
FINAL SCORES
============================================================
  easy      : 0.8230 (Paid: Rs.15)
  medium    : 0.5410 (Paid: Rs.35)
  hard      : 0.2150 (Paid: Rs.80)
  average   : 0.5263
============================================================
```

## Deploying to Hugging Face Spaces

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify a target repo
openenv push --repo-id your-username/park

# Push as private
openenv push --private
```

After deployment your space will be available at:
`https://huggingface.co/spaces/<your-username>/park`

The deployed space includes:
- **Web Interface** at `/web` — Interactive UI for exploring the environment
- **API Documentation** at `/docs` — Full OpenAPI/Swagger interface
- **Health Check** at `/health` — Container health monitoring
- **WebSocket** at `/ws` — Persistent session endpoint for low-latency interactions

## Development & Testing

### Test the Environment Directly (No Server)

```bash
# Test environment logic without starting HTTP server
python3 server/park_environment.py
```

This verifies that:
- Environment resets correctly across all 3 tasks
- Step executes all 5 actions properly
- State tracking and fuel/time limits work
- Rewards are calculated correctly
- `grade()` returns a value in 0.0–1.0

### Run Tests

```bash
# From project root
python3 -m pytest tests/ -v
```

## API Reference

### POST `/reset`

Reset the environment and start a new episode.

```json
{
  "task": "easy"
}
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

Get the current environment state without taking an action.

Returns current episode metadata including `episode_id`, `step_count`, `done`, `task`, and `observation`.

### GET `/health`

Health check endpoint. Returns `{"status": "ok"}` when the server is running.

## Project Structure

```
park/
├── Dockerfile                  # Container image definition (root, used by HF Spaces)
├── README.md                   # This file
├── openenv.yaml                # OpenEnv manifest with task definitions
├── pyproject.toml              # Project metadata and dependencies
├── uv.lock                     # Locked dependencies (generated)
├── __init__.py                 # Module exports (ParkAction, ParkObservation, ParkEnv)
├── client.py                   # ParkEnv HTTP client
├── models.py                   # Action, Observation, and Reward Pydantic models
├── inference.py                # Baseline inference script (runs all 3 tasks)
└── server/
    ├── __init__.py             # Server module exports
    ├── park_environment.py     # Core environment logic (step, reset, grade)
    ├── app.py                  # FastAPI application (HTTP + WebSocket endpoints)
    ├── requirements.txt        # Server dependencies
    └── Dockerfile              # Server-only Dockerfile (for local builds)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes (for inference) | Your Hugging Face API token |
| `OPENAI_API_KEY` | Alternative to HF_TOKEN | OpenAI-compatible API key |
| `API_BASE_URL` | No | LLM API base URL (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | No | Model to use for inference (default: `Qwen/Qwen2.5-72B-Instruct`) |