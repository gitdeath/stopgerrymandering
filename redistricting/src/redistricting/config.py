from __future__ import annotations
import json
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Defaults:
crs_epsg: int = 26915
pop_tolerance_ratio: float = 0.005
compactness_threshold: float = 0.20
progress_interval: int = 5000


@dataclass(frozen=True)
class State:
code: str
fips: str
name: str
districts: int


class Settings:
def __init__(self, defaults: Defaults, states: dict[str, State]):
self.defaults = defaults
self.states = states


@staticmethod
def load(path: str | Path) -> 'Settings':
p = Path(path)
data = yaml.safe_load(p.read_text()) if p.suffix in {'.yaml', '.yml'} else json.loads(p.read_text())
d = data.get('defaults', {})
defaults = Defaults(
crs_epsg=d.get('crs_epsg', 26915),
pop_tolerance_ratio=d.get('pop_tolerance_ratio', 0.005),
compactness_threshold=d.get('compactness_threshold', 0.20),
progress_interval=d.get('progress_interval', 5000),
)
states = {}
for code, rec in data.get('states', {}).items():
states[code.upper()] = State(code=code.upper(), fips=str(rec['fips']), name=rec['name'], districts=int(rec['districts']))
return Settings(defaults, states)
