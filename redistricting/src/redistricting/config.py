from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import json
import yaml


@dataclass(frozen=True)
class Defaults:
    """Global defaults controlled via config (YAML/JSON)."""
    crs_epsg: int = 26915
    pop_tolerance_ratio: float = 0.005
    compactness_threshold: float = 0.20
    progress_interval: int = 5000


@dataclass(frozen=True)
class State:
    """State metadata record."""
    code: str
    fips: str
    name: str
    districts: int


class Settings:
    """
    Container for parsed settings.
    - defaults: global tuning knobs
    - states: map[STATE_CODE] -> State
    """
    def __init__(self, defaults: Defaults, states: Dict[str, State]):
        self.defaults = defaults
        self.states = states

    @staticmethod
    def load(path: str | Path) -> "Settings":
        """
        Load configuration from YAML or JSON.

        Layout example (YAML):
        ---
        defaults:
          crs_epsg: 26915
          pop_tolerance_ratio: 0.005
          compactness_threshold: 0.20
          progress_interval: 5000
        states:
          MO: { fips: '29', name: 'Missouri', districts: 8 }
          CA: { fips: '06', name: 'California', districts: 52 }
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")

        if p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(p.read_text())
        else:
            data = json.loads(p.read_text())

        d = data.get("defaults", {}) or {}
        defaults = Defaults(
            crs_epsg=int(d.get("crs_epsg", 26915)),
            pop_tolerance_ratio=float(d.get("pop_tolerance_ratio", 0.005)),
            compactness_threshold=float(d.get("compactness_threshold", 0.20)),
            progress_interval=int(d.get("progress_interval", 5000)),
        )

        states_raw = data.get("states", {}) or {}
        states: Dict[str, State] = {}
        for code, rec in states_raw.items():
            code_u = code.upper()
            states[code_u] = State(
                code=code_u,
                fips=str(rec["fips"]),
                name=str(rec["name"]),
                districts=int(rec["districts"]),
            )

        if not states:
            raise ValueError("No states found in config. Populate the 'states' section.")

        return Settings(defaults, states)
