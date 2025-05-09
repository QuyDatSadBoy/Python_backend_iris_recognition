from pydantic import BaseModel
from typing import Optional

class DetectEyeData(BaseModel):
    """Entity tương ứng với tblDetectEyeData"""
    id: Optional[int] = None
    imageLink: str
    labelLink: str
    tblDetectEyeDataTrainId: int
    
    class Config:
        from_attributes = True