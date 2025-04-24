from pydantic import BaseModel
from typing import Optional, List


class DetectEyeDataTrain(BaseModel):
    """
    Entity representing the tblDetectEyeDataTrain table
    """
    id: Optional[int] = None
    dataTrainPath: str
    detailFilePath: str
    

class DetectEyeData(BaseModel):
    """
    Entity representing the tblDetectEyeData table
    """
    id: Optional[int] = None
    imageLink: str
    labelLink: str
    tblDetectEyeDataTrainId: int