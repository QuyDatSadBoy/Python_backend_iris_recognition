from fastapi import APIRouter, HTTPException, Form
from typing import List, Optional
from dao.TrainDetectionHistoryDAO import TrainDetectionHistoryDAO
from dao.EyeDetectionModelDAO import EyeDetectionModelDAO
from dao.DetectEyeDataTrainDAO import DetectEyeDataTrainDAO
from entity.TrainDetectionHistory import TrainDetectionHistory
from entity.EyeDetectionModel import EyeDetectionModel
from ultralytics import YOLO
import os
import shutil
import torch
from datetime import datetime

router = APIRouter(prefix="/training", tags=["TrainDetectionHistory"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

@router.get("/")
async def getAllTrainDetectionHistory() -> List[TrainDetectionHistory]:
    """GET /training - Lấy tất cả training history"""
    return TrainDetectionHistoryDAO.getAllTrainDetectionHistory()

@router.get("/{id}")
async def getTrainDetectionHistoryById(id: int) -> TrainDetectionHistory:
    """GET /training/{id} - Lấy training history theo ID"""
    history = TrainDetectionHistoryDAO.getTrainDetectionHistoryById(id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Training history with ID {id} not found")
    return history

@router.post("/")
async def startTraining(
    epochs: int = Form(...),
    batchSize: int = Form(...),
    imageSize: int = Form(...),
    learningRate: float = Form(...),
    detectEyeDataTrainId: int = Form(...)
) -> TrainDetectionHistory:
    """POST /training - Bắt đầu training"""
    
    # Kiểm tra dataset tồn tại
    dataset = DetectEyeDataTrainDAO.getDetectEyeDataTrainById(detectEyeDataTrainId)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {detectEyeDataTrainId} not found")
    
    # Tạo training history record
    train_history = TrainDetectionHistory(
        epochs=epochs,
        batchSize=batchSize,
        imageSize=imageSize,
        learningRate=learningRate,
        tblDetectEyeDataTrainId=detectEyeDataTrainId,
        tblEyeDetectionModelId=None  # Sẽ cập nhật sau khi train xong
    )
    
    history_id = TrainDetectionHistoryDAO.createTrainDetectionHistory(train_history)
    
    # Sử dụng model ID 1 làm pretrained
    pretrained_model = EyeDetectionModelDAO.getEyeDetectionModelById(1)
    if pretrained_model:
        pretrained_weights = os.path.join(BASE_DIR, pretrained_model.modelLink)
    else:
        pretrained_weights = 'yolov8n.pt'
    
    try:
        # Load model
        model = YOLO(pretrained_weights)
        
        # Train model
        results = model.train(
            data=dataset.detailFilePath,
            epochs=epochs,
            batch=batchSize,
            imgsz=imageSize,
            lr0=learningRate,
            device='0' if torch.cuda.is_available() else 'cpu'
        )
        
        # Lấy đường dẫn model đã train
        save_dir = getattr(results, 'save_dir', None)
        if save_dir:
            best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        else:
            # Fallback
            run_dirs = glob.glob(os.path.join('runs', 'detect', 'train*'))
            if run_dirs:
                latest_run = max(run_dirs, key=os.path.getmtime)
                best_model_path = os.path.join(latest_run, 'weights', 'best.pt')
            else:
                raise Exception("Could not find model weights")
        
        # Lưu model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"eye_detector_{timestamp}"
        model_filename = f"{model_name}.pt"
        model_path = os.path.join(MODEL_DIR, model_filename)
        
        shutil.copy(best_model_path, model_path)
        
        # Tạo model record
        new_model = EyeDetectionModel(
            modelName=model_name,
            mapMetric=getattr(results, 'metrics', {}).get('mAP50-95(B)', 0.0),
            isActive=0,
            modelLink=os.path.join("models", model_filename)
        )
        
        model_id = EyeDetectionModelDAO.createEyeDetectionModel(new_model)
        
        # Cập nhật training history với model ID
        train_history.tblEyeDetectionModelId = model_id
        train_history.id = history_id
        
        # Cập nhật trong database
        connection = get_connection()
        try:
            with connection.cursor() as cursor:
                sql = """
                UPDATE tblTrainDetectionHistory 
                SET tblEyeDetectionModelId = %s
                WHERE id = %s
                """
                cursor.execute(sql, (model_id, history_id))
                connection.commit()
        finally:
            connection.close()
        
        return train_history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/{id}/activate")
async def activateTrainedModel(id: int) -> bool:
    """POST /training/{id}/activate - Kích hoạt model từ training history"""
    history = TrainDetectionHistoryDAO.getTrainDetectionHistoryById(id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Training history with ID {id} not found")
    
    if not history.tblEyeDetectionModelId:
        raise HTTPException(status_code=400, detail="No model associated with this training history")
    
    success = EyeDetectionModelDAO.setActiveEyeDetectionModel(history.tblEyeDetectionModelId)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to activate model")
    
    return success