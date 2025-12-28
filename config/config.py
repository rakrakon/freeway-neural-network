from pydantic import BaseModel, Field
from typing import Any
import yaml

class EnvConfig(BaseModel):
    width: int
    height: int
    num_lanes: int
    fps: int

class TrainConfig(BaseModel):
    frame_stack: int
    frame_height: int
    frame_width: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: int
    batch_size: int
    replay_buffer_size: int
    target_update_frequency: int
    learning_starts: int

class Config(BaseModel):
    env: EnvConfig
    train: TrainConfig


with open("config/config.yaml") as f:
    data = yaml.safe_load(f)

cfg = Config(**data)