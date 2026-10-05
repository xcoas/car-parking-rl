from pydantic import BaseModel, Field
from typing import Optional, List

class ModelInfoResponse(BaseModel):
    available_models: List[str]
    observation_space_size: int
    lidar_rays: int
    action_space: List[str]
    network_architecture: dict

class Simulation_Request(BaseModel):
    model_version: str = Field(
        default="v1",
        description="select model version: 'v0' or 'v1'"
    )
    difficulty: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="difficulty parking level from 0.0 to 1.0"
    )
    seed: Optional[int] = Field(
        default=42,
        description="seed of map randomization"
    )

class SimulationResponse(BaseModel):
    model_used: str
    difficulty: float
    seed: int
    is_success: bool
    crashed: bool
    steps_taken: int
    total_reward: float
    initial_critic_value: float
    final_critic_value: float

class CompareRequest(BaseModel):
    difficulty: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="difficulty parking level",
    )
    episodes: int = Field(
        default=10,
        ge=1,
        le=100,
        description="number of comparison games"
    )

class ModelStats(BaseModel):
    win_rate_percent: float
    crashes: int
    avg_steps_on_success: float

class CompareResponse(BaseModel):
    difficulty: float
    episodes_played: int
    winner: str
    model_v0_stats: ModelStats
    model_v1_stats: ModelStats