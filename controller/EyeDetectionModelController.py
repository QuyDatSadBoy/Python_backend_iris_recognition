from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, List
from dao.EyeDetectionModelDAO import EyeDetectionModelDAO
from entity.EyeDetectionModel import EyeDetectionModel
from datetime import datetime
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

router = APIRouter(prefix="/models", tags=["EyeDetectionModel"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = "/home/quydat09/iris_rcog/eye-recognition-system/uploads"

@router.get("/")
async def getAllEyeDetectionModel() -> List[EyeDetectionModel]:
    """GET /models - Lấy tất cả model"""
    return EyeDetectionModelDAO.getAllEyeDetectionModel()

@router.get("/{id}")
async def getEyeDetectionModelById(id: int) -> EyeDetectionModel:
    """GET /models/{id} - Lấy model theo ID"""
    model = EyeDetectionModelDAO.getEyeDetectionModelById(id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model with ID {id} not found")
    return model

@router.post("/")
async def createEyeDetectionModel(
    modelName: str = Form(...),
    modelFile: UploadFile = File(...),
    isActive: Optional[int] = Form(0)
) -> EyeDetectionModel:
    """POST /models - Tạo model mới"""
    # Save model file
    model_filename = f"{modelName}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(modelFile.file, buffer)
    
    # Create model entity
    model = EyeDetectionModel(
        modelName=modelName,
        modelLink=os.path.join("models", model_filename),
        isActive=isActive,
        mapMetric=None
    )
    
    model_id = EyeDetectionModelDAO.createEyeDetectionModel(model)
    
    # If active, deactivate others
    if isActive == 1:
        EyeDetectionModelDAO.setActiveEyeDetectionModel(model_id)
    
    model.id = model_id
    return model

@router.put("/{id}")
async def updateEyeDetectionModel(
    id: int,
    eyeDetectionModel: EyeDetectionModel
) -> EyeDetectionModel:
    """PUT /models/{id} - Cập nhật model"""
    success = EyeDetectionModelDAO.updateEyeDetectionModel(id, eyeDetectionModel)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model with ID {id} not found")
    
    eyeDetectionModel.id = id
    return eyeDetectionModel

@router.delete("/{id}")
async def deleteEyeDetectionModel(id: int) -> bool:
    """DELETE /models/{id} - Xóa model"""
    success = EyeDetectionModelDAO.deleteEyeDetectionModel(id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model with ID {id} not found")
    return success

@router.post("/detect")
async def detectEyesImage(
    image: UploadFile = File(...),
    modelId: Optional[int] = Form(None)
) -> dict:
    """POST /models/detect - Detect eyes trong ảnh"""
    model_info = EyeDetectionModelDAO.getEyeDetectionModelById(modelId)

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    model_path = os.path.join(BASE_DIR, model_info.modelLink)
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    results = model(img)
    
    annotated_img = results[0].plot(conf=0.6)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    face_img_filename = f"face_{timestamp}.jpg"
    face_img_path = os.path.join(UPLOADS_DIR, "faces", face_img_filename)
    
    cv2.imwrite(face_img_path, annotated_img)
    
    eye_regions = []
    
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
        class_name = results[0].names[class_id] if class_id in results[0].names else "unknown"
        
        if confidence > 0.5:
            eye_regions.append({
                "id": i,
                "confidence": confidence,
                "class": class_id,
                "name": class_name,
                "coordinates": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "image_path": ""
            })
            
            if class_name.lower() == "iris" and x1 < x2 and y1 < y2 and x2 <= img.shape[1] and y2 <= img.shape[0]:
                eye_img = img[y1:y2, x1:x2]
                
                if eye_img.size > 0:
                    eye_img_filename = f"eye_{timestamp}_{i}.jpg"
                    eye_img_path = os.path.join(UPLOADS_DIR, "eyes", eye_img_filename)
                    
                    cv2.imwrite(eye_img_path, eye_img)
                    eye_regions[-1]["image_path"] = f"/uploads/eyes/{eye_img_filename}"
    
    return {
        "detected_eyes": eye_regions,
        "annotated_image": f"/uploads/faces/{face_img_filename}",
        "model_used": {
            "id": model_info.id,
            "name": model_info.modelName
        }
    }

@router.post("/batch-detect")
async def batchDetectEyes(
    images: List[UploadFile] = File(...),
    modelId: Optional[int] = Form(None)
) -> dict:
    """POST /models/batch-detect - Batch detect eyes"""
    pass