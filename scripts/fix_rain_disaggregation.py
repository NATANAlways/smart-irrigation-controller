"""Replace the unreliable NASA POWER *hourly* precipitation column with a
physically-realistic hourly series derived from the validated NASA POWER
*daily* precipitation total.

Why: NASA POWER's hourly PRECTOTCORR is known to be unreliable — it produced
values like 40+ mm/hour sustained for 10+ hours, giving an implied annual
total of ~35,000 mm/year for Jaffna (real value: ~700-900 mm/year). This
broke the irrigation simulation's rain model, pinning soil moisture at
field-capacity for weeks and killing the plant via "waterlogging" regardless
of agent behaviour.

Fix: keep NASA's *daily* PRECTOTCORR (the validated, recommended product),
and disaggregate each day's total down into a handful of hours using a
Gaussian (bell curve) profile centred on a randomly chosen peak hour. The
Gaussian weights are normalised so they sum to exactly the daily total — no
rain is invented or lost, only reshaped into realistic hourly bursts.

All other columns (temperature, humidity, wind, solar, et0) are left
untouched — only rain_mm and is_raining are rebuilt.

Usage:
    python scripts/fix_rain_disaggregation.py
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

LAT = 9.7432
LON = 80.0076

NASA_POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

PROJECT_ROOT = Path(__file__).parent.parent
HOURLY_CSV   = PROJECT_ROOT / "data" / "weather" / "uduvil_per_hour" / "uduvil_hourly_2004_2024.csv"
BACKUP_CSV   = PROJECT_ROOT / "data" / "weather" / "uduvil_per_hour" / "uduvil_hourly_2004_2024.original_backup.csv"
DAILY_RAW    = PROJECT_ROOT / "data" / "weather" / "uduvil_per_hour" / "raw" / "daily_prectotcorr_2004_2024.json"

RAIN_THRESHOLD_MM = 0.5  # is_raining flag threshold after disaggregation


def fetch_daily_precip(start_year: int, end_year: int) -> dict[str, float]:
    """Fetch NASA POWER *daily* PRECTOTCORR for the whole range in one call."""
    if DAILY_RAW.exists():
        print(f"Using cached daily precipitation: {DAILY_RAW}")
        with open(DAILY_RAW) as f:
            return json.load(f)

    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": LON,
        "latitude": LAT,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }
    print(f"Fetching NASA POWER daily PRECTOTCORR {start_year}-{end_year}...")
    response = requests.get(NASA_POWER_DAILY_URL, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()["properties"]["parameter"]["PRECTOTCORR"]

    DAILY_RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(DAILY_RAW, "w") as f:
        json.dump(data, f)
    return data


def gaussian_disaggregate(
    daily_total_mm: float,
    rng: random.Random,
    min_sigma: float = 1.0,
    max_sigma: float = 3.0,
    window_sigmas: float = 2.0,
) -> np.ndarray:
    """Spread one day's rain total across 24 hours using a bell curve.

    Returns a (24,) array of hourly rain_mm that sums to exactly
    daily_total_mm. Only hours within `window_sigmas` of the peak get any
    rain — this keeps each rain event short and realistic instead of
    smearing tiny amounts across the whole day.
    """
    hours = np.zeros(24)
    if daily_total_mm <= 0:
        return hours

    peak = rng.uniform(0, 24)
    sigma = rng.uniform(min_sigma, max_sigma)
    window = window_sigmas * sigma

    weights = np.zeros(24)
    for h in range(24):
        # circular distance so a peak near midnight wraps correctly
        dist = abs(h - peak)
        dist = min(dist, 24 - dist)
        if dist <= window:
            weights[h] = math.exp(-(dist ** 2) / (2 * sigma ** 2))

    total_weight = weights.sum()
    if total_weight <= 0:
        # window too narrow (shouldn't normally happen) — put it all in the peak hour
        weights[round(peak) % 24] = 1.0
        total_weight = 1.0

    hours = weights / total_weight * daily_total_mm
    return hours


def main(seed: int = 42) -> None:
    if not HOURLY_CSV.exists():
        raise FileNotFoundError(f"{HOURLY_CSV} not found — run fetch_weather_hourly.py first")

    rng = random.Random(seed)

    print(f"Loading existing hourly CSV: {HOURLY_CSV}")
    df = pd.read_csv(HOURLY_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    start_year = df["datetime"].dt.year.min()
    end_year = df["datetime"].dt.year.max()

    daily_precip = fetch_daily_precip(start_year, end_year)

    if not BACKUP_CSV.exists():
        print(f"Backing up original (broken-rain) CSV to {BACKUP_CSV}")
        df.to_csv(BACKUP_CSV, index=False)
    else:
        print(f"Backup already exists at {BACKUP_CSV} — not overwriting.")

    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    new_rain = np.zeros(len(df))

    dates = sorted(df["date"].unique())
    print(f"Disaggregating {len(dates):,} days of validated daily rainfall into hourly bursts...")

    for date in dates:
        key = date.strftime("%Y%m%d")
        total = daily_precip.get(key, 0.0)
        if total is None or total < -900:  # NASA missing-value flag
            total = 0.0
        total = max(0.0, float(total))

        hourly_profile = gaussian_disaggregate(total, rng)

        day_mask = df["date"] == date
        day_idx = df.index[day_mask]
        hours_for_day = df.loc[day_idx, "hour"].to_numpy()
        new_rain[day_idx] = hourly_profile[hours_for_day]

    df["rain_mm"] = np.round(new_rain, 2)
    df["is_raining"] = (df["rain_mm"] > RAIN_THRESHOLD_MM).astype(int)
    df = df.drop(columns=["date", "hour"])

    df.to_csv(HOURLY_CSV, index=False)

    print(f"\nDone. Rewrote: {HOURLY_CSV}")
    print(f"Original (broken) version preserved at: {BACKUP_CSV}")

    print("\n--- Sanity check ---")
    daily_totals = df.set_index("datetime")["rain_mm"].resample("D").sum()
    print(f"Mean daily rainfall: {daily_totals.mean():.2f} mm/day  (was ~81.5 mm/day before fix)")
    print(f"Fraction of days with any rain: {(daily_totals > 0).mean()*100:.1f}%  (was 92.7% before fix)")
    print(f"Annual average: {daily_totals.mean()*365:.0f} mm/year  (real Jaffna ~700-900 mm/year)")

    df2 = df.copy()
    block = (df2["is_raining"] != df2["is_raining"].shift()).cumsum()
    streaks = df2[df2["is_raining"] == 1].groupby(block[df2["is_raining"] == 1]).size()
    print(f"Rain streak lengths — median: {streaks.median():.1f}h, max: {streaks.max()}h  (was median 11h, max 1055h before fix)")
    print(f"Streaks >= 12h (lethal for Stage-0 wet_hours_limit): {(streaks >= 12).sum()} / {len(streaks)} ({(streaks>=12).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
