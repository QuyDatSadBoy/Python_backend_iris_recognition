from fastapi import APIRouter, HTTPException
from typing import List
from dao.DetectEyeDataDAO import DetectEyeDataDAO
from entity.DetectEyeData import DetectEyeData

router = APIRouter(prefix="/detect-data", tags=["DetectEyeData"])

@router.get("/{detectEyeDataTrainId}")
async def getAllDetectEyeData(detectEyeDataTrainId: int) -> List[DetectEyeData]:
    """GET /detect-data/{detectEyeDataTrainId} - Lấy eye data theo training data ID"""
    return DetectEyeDataDAO.getAllDetectEyeData(detectEyeDataTrainId)

@router.get("/detail/{id}")
async def getDetectEyeDataById(id: int) -> DetectEyeData:
    """GET /detect-data/detail/{id} - Lấy eye data theo ID"""
    data = DetectEyeDataDAO.getDetectEyeDataById(id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Detect eye data with ID {id} not found")
    return data

@router.post("/")
async def createDetectEyeData(detectEyeData: DetectEyeData) -> DetectEyeData:
    """POST /detect-data - Tạo eye data mới"""
    data_id = DetectEyeDataDAO.createDetectEyeData(detectEyeData)
    detectEyeData.id = data_id
    return detectEyeData

@router.delete("/{id}")
async def deleteDetectEyeData(id: int) -> bool:
    """DELETE /detect-data/{id} - Xóa eye data"""
    success = DetectEyeDataDAO.deleteDetectEyeData(id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Detect eye data with ID {id} not found")
    return success