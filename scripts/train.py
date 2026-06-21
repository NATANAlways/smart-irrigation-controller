"""PPO training script for the smart irrigation controller.

Curriculum Learning:
    Phase 1 (40% of timesteps) — Dry season months (Feb–Apr, Jaffna)
             Agent learns core skill: moisture drops → irrigate correctly
    Phase 2 (60% of timesteps) — All months randomly
             Agent generalises to monsoon, moderate, and dry conditions

Usage:
    python scripts/train.py
    python scripts/train.py --timesteps 1000000 --n-envs 4
    python scripts/train.py --output-dir runs/experiment_1
    python scripts/train.py --phase1-ratio 0.3
    python scripts/train.py --log-interval 5
"""



from __future__ import annotations

import wandb
from wandb.integration.sb3 import WandbCallback

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from callbacks import IrrigationMonitorCallback
from irrigation.config_loader import load_config
from irrigation.rl.gym_env import IrrigationGymEnv
from irrigation.zone_config import ZoneConfig

# Load all training parameters from config.yaml
# CLI arguments override these when provided
_cfg   = load_config()
_train = _cfg["training"]
_ppo   = _cfg["ppo"]
_sac   = _cfg["sac"]


def _build_model(algo: str, env, output_path: Path):
    """Construct a fresh PPO or SAC model.

    SAC is off-policy and keeps a replay buffer of past transitions that it
    samples from on every gradient step, instead of PPO's on-policy rollout
    buffer which is discarded after each update.
    """
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            learning_rate  = _ppo["learning_rate"],
            n_steps        = _ppo["n_steps"],
            batch_size     = _ppo["batch_size"],
            n_epochs       = _ppo["n_epochs"],
            gamma          = _ppo["gamma"],
            gae_lambda     = _ppo["gae_lambda"],
            clip_range     = _ppo["clip_range"],
            verbose        = 1,
            tensorboard_log= str(output_path / "logs"),
        )
    if algo == "sac":
        return SAC(
            "MlpPolicy",
            env,
            learning_rate    = _sac["learning_rate"],
            buffer_size      = _sac["buffer_size"],
            learning_starts  = _sac["learning_starts"],
            batch_size       = _sac["batch_size"],
            train_freq       = _sac["train_freq"],
            gradient_steps   = _sac["gradient_steps"],
            gamma            = _sac["gamma"],
            tau              = _sac["tau"],
            verbose          = 1,
            tensorboard_log  = str(output_path / "logs"),
        )
    raise ValueError(f"Unknown algorithm: {algo!r} (expected 'ppo' or 'sac')")


def _make_callbacks(
    output_path: Path,
    eval_env,
    log_interval: int,
    algo: str,
) -> list:
    return [
        EvalCallback(
            eval_env,
            best_model_save_path=str(output_path / "best"),
            eval_freq=10_000,
            n_eval_episodes=5,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(output_path / "checkpoints"),
            name_prefix=f"{algo}_irrigation",
        ),
        IrrigationMonitorCallback(log_interval=log_interval, verbose=1),
        WandbCallback(verbose=0), 
    ]


def train(
    total_timesteps: int = _train["total_timesteps"],
    n_envs: int           = _train["n_envs"],
    output_dir: str       = "models",
    area_m2: float        = _cfg["zone"]["area_m2"],
    irrigation_type: str  = _cfg["zone"]["irrigation_type"],
    phase1_ratio: float   = _train["phase1_ratio"],
    phase2_ratio: float   = _train["phase2_ratio"],
    phase3_ratio: float   = _train["phase3_ratio"],
    log_interval: int     = _train["log_interval"],
    algo: str             = "ppo",
) -> None:


    zone = ZoneConfig(area_m2=area_m2, irrigation_type=irrigation_type)
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = Path(output_dir) / today
    output_path.mkdir(parents=True, exist_ok=True)

    algo_cfg = _ppo if algo == "ppo" else _sac
    run = wandb.init(
        project="smart-irrigation-controller",
        config={
            "algo": algo,
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "phase1_ratio": phase1_ratio,
            "phase2_ratio": phase2_ratio,
            "phase3_ratio": phase3_ratio,
            "learning_rate": algo_cfg["learning_rate"],
            "gamma": algo_cfg["gamma"],
            "area_m2": area_m2,
            "irrigation_type": irrigation_type,
            **({"n_steps": _ppo["n_steps"], "batch_size": _ppo["batch_size"], "clip_range": _ppo["clip_range"]}
               if algo == "ppo" else
               {"buffer_size": _sac["buffer_size"], "batch_size": _sac["batch_size"], "learning_starts": _sac["learning_starts"]}),
        },
        sync_tensorboard=True,
    )

    phase1_steps = int(total_timesteps * phase1_ratio)
    phase2_steps = int(total_timesteps * phase2_ratio)
    phase3_steps = total_timesteps - phase1_steps - phase2_steps

    print("=" * 62)
    print(f"  {algo.upper()} Smart Irrigation — 3-Phase Curriculum Learning")
    print("=" * 62)
    print(f"  Zone          : {area_m2}m²  {irrigation_type}  ({zone.efficiency*100:.0f}% efficiency)")
    print(f"  Total steps   : {total_timesteps:,}")
    print(f"  Phase 1 steps : {phase1_steps:,}  ({phase1_ratio*100:.0f}%)  Yala season (Jan–Mar start)")
    print(f"  Phase 2 steps : {phase2_steps:,}  ({phase2_ratio*100:.0f}%)  Maha season (Aug–Sep start)")
    print(f"  Phase 3 steps : {phase3_steps:,}  ({phase3_ratio*100:.0f}%)  Both seasons, random year/month")
    print(f"  Episode length: 3,600 steps  (150 days × 24 hr/day)")
    print(f"  Parallel envs : {n_envs}")
    print(f"  Log interval  : every {log_interval} episodes")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Phase 1 — Yala season (Jan/Feb/Mar start, fixed year, sequential months)
    # ------------------------------------------------------------------
    print("\n[PHASE 1] Yala season — dry conditions, Jan/Feb/Mar planting...")

    death_warmup_steps = _train["death_warmup_steps"]
    phase1_kwargs = {"zone": zone, "training_phase": 1, "death_warmup_steps": death_warmup_steps}
    train_env_p1 = make_vec_env(IrrigationGymEnv, n_envs=n_envs, env_kwargs=phase1_kwargs)
    eval_env_p1  = make_vec_env(IrrigationGymEnv, n_envs=1,      env_kwargs=phase1_kwargs)

    model = _build_model(algo, train_env_p1, output_path)

    model.learn(
        total_timesteps=phase1_steps,
        callback=_make_callbacks(output_path / "phase1", eval_env_p1, log_interval, algo),
        reset_num_timesteps=True,
    )

    phase1_save = output_path / f"{algo}_phase1_yala"
    model.save(str(phase1_save))
    print(f"\n[PHASE 1] Complete — model saved to {phase1_save}")

    # ------------------------------------------------------------------
    # Phase 2 — Maha season (Oct/Nov start, monsoon conditions)
    # ------------------------------------------------------------------
    if phase2_steps > 0:
        print("\n[PHASE 2] Maha season — monsoon conditions, Oct/Nov planting...")

        phase2_kwargs = {"zone": zone, "training_phase": 2}
        train_env_p2 = make_vec_env(IrrigationGymEnv, n_envs=n_envs, env_kwargs=phase2_kwargs)
        eval_env_p2  = make_vec_env(IrrigationGymEnv, n_envs=1,      env_kwargs=phase2_kwargs)

        model.set_env(train_env_p2)
        model.learn(
            total_timesteps=phase2_steps,
            callback=_make_callbacks(output_path / "phase2", eval_env_p2, log_interval, algo),
            reset_num_timesteps=False,
        )

        phase2_save = output_path / f"{algo}_phase2_maha"
        model.save(str(phase2_save))
        print(f"\n[PHASE 2] Complete — model saved to {phase2_save}")
    else:
        print("\n[PHASE 2] Skipped (phase2_ratio = 0)")

    # ------------------------------------------------------------------
    # Phase 3 — Both seasons, random year per month (maximum variability)
    # ------------------------------------------------------------------
    if phase3_steps > 0:
        print("\n[PHASE 3] Both seasons — random year per month, full variability...")

        phase3_kwargs = {"zone": zone, "training_phase": 3}
        train_env_p3 = make_vec_env(IrrigationGymEnv, n_envs=n_envs, env_kwargs=phase3_kwargs)
        eval_env_p3  = make_vec_env(IrrigationGymEnv, n_envs=1,      env_kwargs=phase3_kwargs)

        model.set_env(train_env_p3)
        model.learn(
            total_timesteps=phase3_steps,
            callback=_make_callbacks(output_path / "phase3", eval_env_p3, log_interval, algo),
            reset_num_timesteps=False,
        )

        final_save = output_path / f"{algo}_irrigation_final"
        model.save(str(final_save))
        print(f"\n[PHASE 3] Complete — final model saved to {final_save}")
    else:
        print("\n[PHASE 3] Skipped (phase3_ratio = 0)")
    print("\nTraining complete.")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO irrigation agent with curriculum learning."
    )
    parser.add_argument("--timesteps",       type=int,   default=_train["total_timesteps"])
    parser.add_argument("--n-envs",          type=int,   default=_train["n_envs"])
    parser.add_argument("--output-dir",      type=str,   default="models")
    parser.add_argument("--area",            type=float, default=_cfg["zone"]["area_m2"],
                        help="Zone area in m²")
    parser.add_argument("--irrigation-type", type=str,  default=_cfg["zone"]["irrigation_type"],
                        choices=["drip", "sprinkler"])
    parser.add_argument("--phase1-ratio",    type=float, default=_train["phase1_ratio"],
                        help="Fraction of timesteps for Phase 1 — Yala season")
    parser.add_argument("--phase2-ratio",    type=float, default=_train["phase2_ratio"],
                        help="Fraction of timesteps for Phase 2 — Maha season")
    parser.add_argument("--phase3-ratio",    type=float, default=_train["phase3_ratio"],
                        help="Fraction of timesteps for Phase 3 — both seasons")
    parser.add_argument("--log-interval",    type=int,   default=_train["log_interval"],
                        help="Print episode summary every N episodes")
    parser.add_argument("--algo",            type=str,   default="ppo", choices=["ppo", "sac"],
                        help="RL algorithm — 'sac' uses a replay buffer to reuse past transitions")
    args = parser.parse_args()

    train(
        args.timesteps,
        args.n_envs,
        args.output_dir,
        args.area,
        args.irrigation_type,
        args.phase1_ratio,
        args.phase2_ratio,
        args.phase3_ratio,
        args.log_interval,
        args.algo,
    )
