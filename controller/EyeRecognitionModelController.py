import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
import json
import cv2
import numpy as np
import torch
from PIL import Image
import io
from datetime import datetime
from ultralytics import YOLO
from dao.EyeRecognitionModelDAO import EyeRecognitionModelDAO
from entity.EyeRecognitionModel import EyeRecognitionModel

# Define the router
router = APIRouter(prefix="/models", tags=["Models"])

# Path constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOADS_DIR = "/home/quydat09/iris_rcog/eye-recognition-system/uploads"

# Make sure the directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "eyes"), exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "faces"), exist_ok=True)


@router.get("/")
async def get_all_models():
    """
    Get all models
    """
    models = EyeRecognitionModelDAO.get_all_models()
    return {"models": models}


@router.get("/active")
async def get_active_model():
    """
    Get the currently active model
    """
    model = EyeRecognitionModelDAO.get_active_model()
    if not model:
        raise HTTPException(status_code=404, detail="No active model found")
    return {"model": model}


@router.get("/{model_id}")
async def get_model_by_id(model_id: int):
    """
    Get a model by ID
    """
    model = EyeRecognitionModelDAO.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    return {"model": model}


@router.post("/")
async def create_model(
    modelName: str = Form(...),
    modelFile: UploadFile = File(...),
    isActive: Optional[int] = Form(0)
):
    """
    Create a new model by uploading a model file
    """
    # Save the model file
    model_filename = f"{modelName}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    with open(model_path, "wb") as buffer:
        shutil.copyfileobj(modelFile.file, buffer)
    
    # Store relative path in database
    relative_path = os.path.join("models", model_filename)
    
    # Create the model in the database
    model = EyeRecognitionModel(
        modelName=modelName,
        modelLink=relative_path,
        isActive=isActive,
        mapMetric=None  # We don't know the mAP yet
    )
    
    model_id = EyeRecognitionModelDAO.create_model(model)
    
    # If this model should be active, deactivate all others
    if isActive == 1:
        EyeRecognitionModelDAO.set_active_model(model_id)
    
    return {"message": "Model created successfully", "model_id": model_id}


@router.put("/{model_id}/activate")
async def activate_model(model_id: int):
    """
    Set a model as active
    """
    model = EyeRecognitionModelDAO.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    success = EyeRecognitionModelDAO.set_active_model(model_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to activate model")
    
    return {"message": f"Model with ID {model_id} is now active"}


@router.delete("/{model_id}")
async def delete_model(model_id: int):
    """
    Delete a model
    """
    model = EyeRecognitionModelDAO.get_model_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    success = EyeRecognitionModelDAO.delete_model(model_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete model")
    
    return {"message": f"Model with ID {model_id} deleted successfully"}


@router.post("/detect-eyes")
async def detect_eyes(
    image: UploadFile = File(...),
    model_id: Optional[int] = Form(None)
):
    """
    Detect eyes in an image using the specified model or the active model
    """
    # Get the model to use
    if model_id:
        model_info = EyeRecognitionModelDAO.get_model_by_id(model_id)
    else:
        model_info = EyeRecognitionModelDAO.get_active_model()
    
    if not model_info:
        raise HTTPException(status_code=404, detail="No model found")
    
    # Load the image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    # Convert relative model path to absolute
    model_path = os.path.join(BASE_DIR, model_info['modelLink'])
    
    # Load the YOLOv11 model
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    # Perform inference
    results = model(img)
    
    # Save the annotated image
    annotated_img = results[0].plot(conf=0.6)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    face_img_filename = f"face_{timestamp}.jpg"
    face_img_path = os.path.join(UPLOADS_DIR, "faces", face_img_filename)
    
    cv2.imwrite(face_img_path, annotated_img)
    
    # Extract and save detected eye regions
    eye_regions = []
    
    # Process YOLO results correctly
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
        class_name = results[0].names[class_id] if class_id in results[0].names else "unknown"
        
        if confidence > 0.6:  # Confidence threshold
            # Crop the eye region
            if class_name.lower() == "iris" and x1 < x2 and y1 < y2 and x2 <= img.shape[1] and y2 <= img.shape[0]:
                eye_img = img[y1:y2, x1:x2]
                
                # Check if image is valid
                if eye_img.size > 0:
                    # Save the eye image
                    eye_img_filename = f"eye_{timestamp}_{i}.jpg"
                    eye_img_path = os.path.join(UPLOADS_DIR, "eyes", eye_img_filename)
                    
                    cv2.imwrite(eye_img_path, eye_img)
                    
                    # Add to the list of detected eye regions
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
                        "image_path": f"/uploads/eyes/{eye_img_filename}"
                    })
    
    return {
        "detected_eyes": eye_regions,
        "annotated_image": f"/uploads/faces/{face_img_filename}",
        "model_used": {
            "id": model_info['id'],
            "name": model_info['modelName']
        }
    }


@router.post("/batch-detect")
async def batch_detect_eyes(
    images: List[UploadFile] = File(...),
    model_id: Optional[int] = Form(None)
):
    """
    Detect eyes in multiple images
    """
    # Get the model to use
    if model_id:
        model_info = EyeRecognitionModelDAO.get_model_by_id(model_id)
    else:
        model_info = EyeRecognitionModelDAO.get_active_model()
    
    if not model_info:
        raise HTTPException(status_code=404, detail="No model found")
    
    # Convert relative model path to absolute
    model_path = os.path.join(BASE_DIR, model_info['modelLink'])
    
    # Load the YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    results = []
    for image in images:
        try:
            # Load the image
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                # Skip invalid images
                results.append({
                    "filename": image.filename,
                    "error": "Invalid image",
                    "detected_eyes": [],
                    "annotated_image": None
                })
                continue
            
            # Perform inference
            detection_results = model(img)
            
            # Save the annotated image
            annotated_img = detection_results[0].plot()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in image.filename)
            face_img_filename = f"face_{safe_filename}_{timestamp}.jpg"
            face_img_path = os.path.join(UPLOADS_DIR, "faces", face_img_filename)
            
            cv2.imwrite(face_img_path, annotated_img)
            
            # Extract and save detected eye regions
            eye_regions = []
            
            # Process YOLO results correctly
            for i, box in enumerate(detection_results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                class_name = detection_results[0].names[class_id] if class_id in detection_results[0].names else "unknown"
                
                # Thêm tất cả các vùng phát hiện vào danh sách (không phụ thuộc vào class)
                if confidence > 0.6:  # Confidence threshold
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
                        "image_path": ""  # Mặc định là rỗng, sẽ cập nhật nếu là iris
                    })
                    
                    # Chỉ lưu ảnh cắt khi class_name là "iris"
                    if class_name.lower() == "iris" and x1 < x2 and y1 < y2 and x2 <= img.shape[1] and y2 <= img.shape[0]:
                        eye_img = img[y1:y2, x1:x2]
                        
                        # Check if image is valid
                        if eye_img.size > 0:
                            # Save the eye image
                            eye_img_filename = f"eye_{safe_filename}_{timestamp}_{i}.jpg"
                            eye_img_path = os.path.join(UPLOADS_DIR, "eyes", eye_img_filename)
                            
                            cv2.imwrite(eye_img_path, eye_img)
                            
                            # Cập nhật đường dẫn ảnh trong thông tin vùng phát hiện
                            eye_regions[-1]["image_path"] = f"/uploads/eyes/{eye_img_filename}"
            
            results.append({
                "filename": image.filename,
                "detected_eyes": eye_regions,
                "annotated_image": f"/uploads/faces/{face_img_filename}"
            })
        
        except Exception as e:
            results.append({
                "filename": image.filename,
                "error": f"Error processing image: {str(e)}",
                "detected_eyes": [],
                "annotated_image": None
        })
    return {
        "batch_results": results,
        "model_used": {
            "id": model_info['id'],
            "name": model_info['modelName']
        }
    }
    
    
    
    