# System Architecture

## Overview

The smart irrigation controller is a PPO-based reinforcement learning system that
learns to irrigate a Jaffna chilli bed using real historical weather data. One
episode simulates one full 150-day growing season at hourly resolution.

---

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       IrrigationGymEnv                             │
│                    (Gymnasium training wrapper)                     │
│                                                                     │
│  ┌─────────────────────┐      ┌──────────────────────────────────┐ │
│  │  HistoricalWeather  │      │        PPO Agent                 │ │
│  │  Sensor             │      │  (Stable-Baselines3 MlpPolicy)   │ │
│  │  (NASA POWER data   │      │                                  │ │
│  │   2004-2024, Jaffna)│      │  Actor  → continuous action      │ │
│  └────────┬────────────┘      │  Critic → value estimate         │ │
│           │                   └───────────────┬──────────────────┘ │
│           ▼                                   │                     │
│  ┌────────────────────┐   Box(9,) obs         │  Box(1,) action     │
│  │ SimulatedSoil      │ ◀─────────────────────┘                     │
│  │ MoistureSensor     │                                             │
│  │ (ET₀-driven, +rain)│                                             │
│  └────────┬───────────┘                                             │
│           │                   ┌──────────────────────────────────┐ │
│           ▼                   │       IrrigationEnvironment      │ │
│  ┌────────────────────┐       │  observe() → IrrigationState     │ │
│  │ SimulatedActuator  │◀──────│  step(litres) → reward           │ │
│  │ (applies litres to │       │  RewardFunction.compute()        │ │
│  │  soil moisture)    │       └──────────────────────────────────┘ │
│  └────────────────────┘                                             │
│                                                                     │
│  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────────┐ │
│  │ DynamicStage     │  │ PlantVitality     │  │ PlantHealth     │ │
│  │ Tracker          │  │ Tracker           │  │ Tracker         │ │
│  │ (health-conditioned│  │ (death conditions)│  │ (health score   │ │
│  │  stage advance)  │  │                   │  │  in obs)        │ │
│  └──────────────────┘  └───────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow (one step)

```
[HistoricalWeatherSensor]
  └─ reads real NASA POWER hourly record at current replay index
  └─ converts ET₀ (mm/hr) → soil drain rate (%/hr)
  └─ applies rainfall_mm to soil if raining
  └─ reads SimulatedSoilMoistureSensor → SensorReading
         │
         ▼
[IrrigationEnvironment.observe()]
  └─ builds IrrigationState (7 fields, normalized to [0,1])
  └─ day and stage overridden by IrrigationGymEnv (simulated clock)
         │
         ▼
[IrrigationGymEnv._make_obs()]
  └─ appends health_score + vitality → Box(9,) observation
         │
         ▼
[PPO MlpPolicy]
  └─ outputs action: float in [0, 1]
  └─ mapped to water_litres = action × max_litres_per_event
         │
         ▼
[SimulatedActuator.execute()]
  └─ adds water_litres to soil moisture sensor
         │
         ▼
[IrrigationEnvironment.step()]
  └─ RewardFunction.compute() → scalar reward
         │
         ▼
[DynamicStageTracker.update()]   [PlantVitalityTracker.update()]   [PlantHealthTracker.update()]
  └─ may advance stage this step   └─ checks death condition          └─ updates health_score
         │                                │
         │                        plant_dead? → add death_penalty to reward
         │                        warmup active? → reset vitality instead of terminating
         ▼
terminated = (step >= 3600) OR (plant_dead AND warmup complete)
```

---

## Space Definitions

### Observation — `Box(9,)`, all values in `[0, 1]`

| Index | Value | Normalization |
|-------|-------|---------------|
| 0 | Soil moisture % | `/ 100` |
| 1 | Temperature °C | `/ 40` |
| 2 | Humidity % | `/ 100` |
| 3 | Hour of day | `/ 24` |
| 4 | Is raining | `0` or `1` |
| 5 | Growth stage (0–4) | `/ 4` |
| 6 | Current day | `/ 150` |
| 7 | Health score | `[0, 1]` |
| 8 | Plant vitality | `[0, 1]` |

### Action — `Box(1,)`, continuous in `[0, 1]`

Agent output is a fraction mapped to `[0, max_litres_per_event]` (default 2.0 L).
`0.0` = no irrigation. `1.0` = maximum water.

---

## Reward Function (`rl/reward.py`)

Each step reward combines:

| Component | Effect |
|-----------|--------|
| Stage-specific stress penalty | `-stress_penalty_weight × stress_level` |
| Overwatering penalty | `-overwater_penalty × (excess / 10)` if above field capacity |
| Optimal moisture bonus | `+0.5` if moisture in `[optimal_min, optimal_max]` for stage |
| Water conservation penalty | `-water_penalty_weight × (litres / max_litres)` |
| Rain bonus | `+rain_bonus` if not irrigating while raining |
| Emergency penalty | `-5.0` if at wilting point but agent applied < emergency_min_litres |
| Death penalty (terminal) | Stage-weighted: -20 (Stage 0) to -80 (Stage 1, peak investment) |

All weights are set in `config.yaml → reward`.

---

## Growth Stage System

### DynamicStageTracker (`rl/dynamic_stage.py`)

Replaces the old calendar-based stage with a **health-conditioned** model.
Stage advance requires **both**:
1. Minimum hours elapsed in the current stage
2. Health ratio ≥ threshold OR maximum hours reached (forced/stunted advance)

Health ratio per step:
- Moisture in optimal range → `+1.0`
- Moisture in stress range → `+0.5`
- Below stress threshold → `+0.0`

| Stage | Name | Min days | Max days | Health required |
|-------|------|----------|----------|----------------|
| 0 | Germination | 15 | 30 | 70% |
| 1 | Vegetative | 30 | 55 | 70% |
| 2 | Flowering | 25 | 40 | 75% (critical) |
| 3 | Fruit Development | 25 | 40 | 70% |
| 4 | Maturity | 25 | 40 | 65% |

### PlantVitalityTracker (`rl/vitality.py`)

Tracks plant survival via stage-specific consecutive dry/wet hour limits.
Vitality drains each stressed hour; reaching 0 = death → episode terminates.

Key agronomic decisions:
- Stage 0 (germination): dies after **2 dry hours** (no root system)
- Stage 2 (flowering): dry threshold is **stress threshold (55%)**, not wilting point — flower drop = crop failure
- Waterlogging kills at half the drain rate of drought (root suffocation is slower)

---

## Curriculum Training (`scripts/train.py`)

Training uses **3 phases** on the same PPO model (weights persist across phases):

| Phase | Season | Timesteps | Purpose |
|-------|--------|-----------|---------|
| 1 | Yala (Jan/Feb/Mar start) | 25% = 125k | Learn basic irrigation in dry, predictable conditions |
| 2 | Maha (Aug/Sep start) | 35% = 175k | Generalize to monsoon — learn rain bonus |
| 3 | Both, random year/month | 40% = 200k | Maximum variability, robust policy |

**Death-disabled warm-up** (Phase 1 only): for the first `death_warmup_steps` (default 20,000)
the plant cannot permanently die — vitality resets on death so the random policy can
experience all 5 growth stages before dying early becomes a training obstacle.

**Episode length**: 150 days × 24 hr/step = **3,600 steps per episode**

**Parallelism**: 4 vectorized environments via `make_vec_env`

---

## Weather Data (`sensors/historical.py`, `weather/weather_data.py`)

- Source: NASA POWER hourly data, Uduvil, Jaffna, **2004–2024** (≈ 175,000 records)
- Each episode picks a **random contiguous start point** in the history, then walks
  forward step-by-step — not independent samples. This preserves real day-to-day
  weather persistence (multi-day rain events, gradual drying trends, diurnal cycles).
- ET₀ from each record drives the soil moisture drain rate:
  `drain %/hr = ET₀_mm / (root_depth_m × 10)`
- Rainfall is applied directly to the soil sensor each step.

---

## Module Descriptions

### `irrigation.config_loader`
Loads and validates `config.yaml`. All components read their parameters from here —
no hardcoded values in source files.

### `irrigation.sensors`
- `SensorReading` — dataclass for one timestep snapshot
- `SimulatedSoilMoistureSensor` — physics-based moisture model; tracks ET₀ drain + irrigation additions
- `HistoricalWeatherSensor` — replays the NASA POWER CSV; updates soil sensor with real ET₀ and rainfall each step
- `SoilMoistureSensor` / `WeatherSensor` — real hardware interfaces (ADS1115 ADC, DHT22)

### `irrigation.actuators`
- `IrrigationCommand` — carries `water_litres` (continuous float)
- `ActuatorInterface` — abstract base; `execute(command)`, `stop()`, `is_active`
- `SimulatedActuator` — adds water to the soil sensor; records events
- `ValveActuator` — GPIO relay driver for the physical solenoid valve

### `irrigation.crops`
- `CropProfile` — abstract base; `moisture_thresholds_for_stage(stage)`, `stress_level_for_stage(moisture, stage)`
- `ChiliProfile` — concrete profile for *Capsicum annuum* (Jaffna chilli); 5 growth stages with FAO Paper 56 thresholds

### `irrigation.rl`
- `IrrigationEnvironment` — inner environment; `observe() → IrrigationState`, `step(litres) → (state, reward, done)`
- `IrrigationGymEnv` — Gymnasium wrapper (`gymnasium.Env`); action/observation spaces, episode management, tracker coordination
- `RewardFunction` — computes per-step scalar reward
- `DynamicStageTracker` — health-conditioned stage progression
- `PlantVitalityTracker` — stage-specific death conditions + terminal penalty
- `PlantHealthTracker` — running health score (proportion of hours in optimal moisture range)
- `QLearningAgent` — legacy tabular Q-learning agent; superseded by PPO, kept for reference

### `irrigation.zone_config`
`ZoneConfig` — zone area, irrigation type (drip/sprinkler), efficiency factor,
moisture-per-litre conversion, and litres limits.

### `irrigation.controller`
Top-level orchestrator wiring all components for production use (`train()`, `decide()`, `run()`).

### `irrigation.cli`
Click CLI: `run`, `train`, `generate-config` sub-commands.

### `scripts/train.py`
PPO training entry point. Reads all hyperparameters from `config.yaml`. Saves
per-phase checkpoints and a final model. Logs to wandb + TensorBoard.

### `scripts/callbacks.py`
`IrrigationMonitorCallback` — SB3 callback that logs per-episode agronomic metrics
(water used, health score, stage reached, death cause) to wandb.

---

## Design Principles

1. **Config-driven** — every threshold, weight, and hyperparameter lives in `config.yaml`. No magic numbers in source files.
2. **Hardware independence** — every hardware component (`SoilMoistureSensor`, `WeatherSensor`, `ValveActuator`) has a simulated counterpart for offline training.
3. **Real-world weather** — training uses 20 years of real NASA POWER hourly data, not a synthetic model, so the learned policy must handle real seasonal patterns.
4. **Health-conditioned stages** — plant growth responds to care quality, not just elapsed time, making irrigation quality directly visible in the reward signal.
5. **Water savings first** — the reward function is designed to directly optimise the primary project metric: grow a healthy crop with minimum water use.
