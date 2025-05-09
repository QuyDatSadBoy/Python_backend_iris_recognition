from pydantic import BaseModel
from typing import Optional, List

class DetectEyeDataTrain(BaseModel):
    """Entity tương ứng với tblDetectEyeDataTrain"""
    id: Optional[int] = None
    dataTrainPath: str
    detailFilePath: str
    
    class Config:
        from_attributes = True