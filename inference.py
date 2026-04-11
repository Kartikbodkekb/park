
import os
import sys
import textwrap
import json
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── FIX: robust import regardless of WORKDIR inside Docker ───────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.park_environment import ParkEnvironment

# ── MANDATORY CONFIGURATION ───────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or "park-env:latest"  #added or "park-env:latest" this on 11-04 
API_KEY      = os.getenv("HF_TOKEN") 
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK  = os.getenv("MY_ENV_V4_BENCHMARK", "smart_parking_env")
STEP_LIMITS = {"easy": 20, "medium": 30, "hard": 40}

TEMPERATURE = 0.1
MAX_TOKENS  = 50

# ── LOGGING HELPERS (STRICT FORMAT) ──────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── PROMPT BUILDERS ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI driver navigating congested urban parking in Indian cities.
    At each step you receive an observation (JSON).
    Choose exactly ONE action: move_to_nearby, move_to_far, explore_random, wait, leave_area.
    Respond with ONLY the action string. No quotes, no explanation.
""").strip()

def build_user_prompt(step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    obs_json = json.dumps(obs, indent=2)
    return textwrap.dedent(f"""
        Step: {step}
        Current Observation:
        {obs_json}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}

        Send your next action.
    """).strip()

# ── MAIN RUNNER ───────────────────────────────────────────────────────────────
def main() -> None:
    if not API_KEY:
        raise ValueError("HF_TOKEN missing in .env")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = ParkEnvironment.from_docker_image(IMAGE_NAME)

    tasks_to_run = ["easy", "medium", "hard"]
    all_results  = {}

    for current_task in tasks_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {current_task.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        history, rewards = [], []
        steps_taken, last_reward, score, success = 0, 0.0, 0.0, False
        task_max_steps = STEP_LIMITS.get(current_task, 35)

        log_start(current_task, BENCHMARK, MODEL_NAME)

        try:
            # 1. INITIALIZE THE EPISODE (Notice we use .model_dump() here)
            result = env.reset(task=current_task)
            obs    = result.observation.model_dump() 

            for step in range(1, task_max_steps + 1):
                if result.done:
                    break

                # The prompt expects 'obs' to be a dict, which it now is!
                prompt   = build_user_prompt(step, obs, last_reward, history)
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                action = (response.choices[0].message.content or "wait").strip().lower().replace('"', "").replace("'", "")
                if not action:
                    action = "wait"

                # 2. TAKE THE ACTION
                result       = env.step(action)
                reward, done = result.reward, result.done

                rewards.append(reward)
                steps_taken = step
                last_reward = reward
                
                # 3. GET THE NEW OBSERVATION (Notice we use .model_dump() here too)
                obs = result.observation.model_dump() 

                error_msg = obs.get("last_action_result") if done else None
                log_step(step, action, reward, done, error_msg)
                history.append(f"S{step}: {action}({reward:+.1f})")

                if done:
                    break

            # 4. GRADE THE EPISODE
            score   = env.grade()
            success = obs.get("parked", False)
            price_paid = obs.get("price_paid", 0.0)
            all_results[current_task] = {"score": score, "price": price_paid}

        except Exception as e:
            print(f"[ERROR] task={current_task} error={e}", flush=True)
            all_results[current_task] = {"score": 0.0, "price": 0.0}
        finally:
            log_end(success, steps_taken, score, rewards)
    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'='*60}", flush=True)

    total_score = 0
    for task, data in all_results.items():
        print(f"  {task:10s}: {data['score']:.4f} (Paid: Rs.{data['price']:.0f})", flush=True)
        total_score += data["score"]

    avg_score = total_score / len(tasks_to_run)
    print(f"  {'average':10s}: {avg_score:.4f}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        env.close()
    except Exception as e:
        print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    main()