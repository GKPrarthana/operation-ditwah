from pydantic import BaseModel, Field
from typing import Literal, Optional

class CrisisEvent(BaseModel):
    district: Literal["Colombo", "Gampaha", "Kandy", "Kalutara", "Galle", "Kegalle", "Ratnapura", "Matara", "Badulla", "Nuwara Eliya"]
    flood_level_meters: Optional[float] = None
    victim_count: int = Field(default=0)
    main_need: str
    status: Literal["Critical", "Warning", "Stable"]