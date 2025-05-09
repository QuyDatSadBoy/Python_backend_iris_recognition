from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TrainDetectionHistory(BaseModel):
    """Entity tương ứng với tblTrainDetectionHistory"""
    id: Optional[int] = None
    epochs: int
    batchSize: int  
    imageSize: int
    learningRate: float
    timeTrain: Optional[datetime] = None
    tblDetectEyeDataTrainId: int
    tblEyeDetectionModelId: Optional[int] = None
    
    
    class Config:
        from_attributes = True