from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class EyeDetectionModel(BaseModel):
    """Entity tương ứng với tblEyeDetectionModel"""
    id: Optional[int] = None
    modelName: str
    mapMetric: Optional[float] = None  
    createDate: Optional[datetime] = None
    isActive: int = 0
    modelLink: str  # Đường dẫn đến file model
    
    
    class Config:
        from_attributes = True