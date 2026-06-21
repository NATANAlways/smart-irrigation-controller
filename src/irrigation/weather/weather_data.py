"""Weather data loader for Uduvil, Jaffna historical hourly records.

Loads the NASA POWER hourly CSV as one chronologically ordered timeline and
provides random *continuous* starting points for episodes — so an episode
replays a real, contiguous slice of history (with realistic day-to-day
persistence: gradual drying trends, multi-day rain events, diurnal cycles)
instead of resampling an independent random hour at every step.

Usage:
    loader = WeatherDataLoader()
    start  = loader.random_start_index(episode_hours=3600, allowed_months=[1, 2, 3])
    record = loader.record_at(start)        # hour 0 of the episode
    record = loader.record_at(start + 1)    # hour 1, etc.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# Default path to the downloaded NASA POWER hourly CSV
_DEFAULT_CSV = (
    Path(__file__).parent.parent.parent.parent
    / "data" / "weather" / "uduvil_per_hour" / "uduvil_hourly_2004_2024.csv"
)

# Jaffna season month groups for curriculum learning
YALA_MONTHS = [1, 2, 3]   # Jan, Feb, Mar — dry season planting window
MAHA_MONTHS = [10, 11]    # Oct, Nov — Maha season actual planting window (post-monsoon onset)


class WeatherRecord(NamedTuple):
    """One hourly weather record from the NASA POWER dataset."""
    temperature:   float   # °C
    humidity_pct:  float   # %
    rain_mm:       float   # mm/hr
    is_raining:    bool    # rain_mm > 0.1
    wind_speed_ms: float   # m/s
    solar_rad_MJ:  float   # MJ/m²/hr
    et0_mm:        float   # mm/hr (FAO-56 Penman-Monteith)


class WeatherDataLoader:
    """Loads the NASA POWER hourly CSV as one continuous chronological timeline.

    Args:
        csv_path: Path to the hourly CSV file. Defaults to the project data folder.
    """

    def __init__(self, csv_path: str | Path | None = None) -> None:
        path = Path(csv_path) if csv_path else _DEFAULT_CSV

        if not path.exists():
            raise FileNotFoundError(
                f"Hourly weather CSV not found at {path}.\n"
                f"Run: python scripts/fetch_weather_hourly.py"
            )

        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        self._months: list[int] = df["datetime"].dt.month.tolist()
        self._records: list[WeatherRecord] = [
            WeatherRecord(
                temperature   = float(row.temperature),
                humidity_pct  = float(row.humidity_pct),
                rain_mm       = float(row.rain_mm),
                is_raining    = bool(row.is_raining),
                wind_speed_ms = float(row.wind_speed_ms),
                solar_rad_MJ  = float(row.solar_rad_MJ),
                et0_mm        = float(row.et0_mm),
            )
            for row in df.itertuples()
        ]

        self._rng = random.Random()
        print(
            f"WeatherDataLoader: loaded {len(self._records):,} records "
            f"({df['datetime'].min().date()} → {df['datetime'].max().date()})"
        )

    def __len__(self) -> int:
        return len(self._records)

    def random_start_index(
        self,
        episode_hours: int,
        allowed_months: list[int] | None = None,
    ) -> int:
        """Pick a random index to start an episode's continuous weather slice.

        Args:
            episode_hours:  Number of hourly steps the episode will run for.
            allowed_months: If given, the start hour's calendar month must be
                             one of these (used for season-based curriculum
                             phases). None = any month.

        Returns:
            An index `i` such that `record_at(i + h)` is valid for all
            `h in [0, episode_hours)` — i.e. the slice never runs past the
            end of the dataset (no wrap-around).
        """
        last_valid = len(self._records) - episode_hours
        if last_valid <= 0:
            raise ValueError("episode_hours is longer than the available data")

        if allowed_months is None:
            return self._rng.randint(0, last_valid - 1)

        candidates = [
            i for i in range(last_valid)
            if self._months[i] in allowed_months
        ]
        if not candidates:
            return self._rng.randint(0, last_valid - 1)
        return self._rng.choice(candidates)

    def record_at(self, index: int) -> WeatherRecord:
        """Return the weather record at the given continuous timeline index."""
        return self._records[index]
