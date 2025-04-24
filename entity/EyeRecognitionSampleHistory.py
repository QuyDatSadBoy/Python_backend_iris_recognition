from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class TrainDetectionHistory(BaseModel):
    """
    Entity representing the tblTrainDetectionHistory table
    """
    id: Optional[int] = None
    epochs: int
    batchSize: int
    imageSize: int
    learningRate: float
    timeTrain: Optional[datetime] = None
    tblDetectEyeDataTrainId: int
    tblEyeDetectionModelId: Optional[int] = None