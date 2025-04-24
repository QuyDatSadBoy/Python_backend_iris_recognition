from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class EyeRecognitionModel(BaseModel):
    """
    Entity representing the tblEyeDetectionModel table
    """
    id: Optional[int] = None
    modelName: str
    mapMetric: Optional[float] = None
    createDate: Optional[datetime] = None
    isActive: Optional[int] = 0
    modelLink: str
    
    class Config:
        arbitrary_types_allowed = True