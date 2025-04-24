import os
import json
import yaml
import torch
import shutil
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import threading
import logging
from io import StringIO
import sys

from dao.EyeRecognitionModelDAO import EyeRecognitionModelDAO
from dao.EyeRecognitionSampleDAO import EyeRecognitionSampleDAO
from dao.EyeRecognitionSampleHistoryDAO import EyeRecognitionSampleHistoryDAO
from entity.EyeRecognitionSampleHistory import TrainDetectionHistory
from entity.EyeRecognitionModel import EyeRecognitionModel

# Define the router
router = APIRouter(prefix="/training", tags=["Training"])

# Path constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

# Make sure the directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Dictionary to store training logs for each training job
training_logs = {}


class LogCapture:
    """Class to capture logs during model training"""
    def __init__(self, job_id):
        self.job_id = job_id
        self.log_stream = StringIO()
        self.old_stdout = None
        training_logs[job_id] = ""

    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout

    def write(self, message):
        self.old_stdout.write(message)
        self.log_stream.write(message)
        training_logs[self.job_id] += message

    def flush(self):
        self.old_stdout.flush()
        self.log_stream.flush()

    def get_logs(self):
        return self.log_stream.getvalue()


def train_yolov11_model(
    dataset_id: int,
    epochs: int,
    batch_size: int,
    image_size: int,
    learning_rate: float,
    pretrained_model_id: Optional[int] = None,
    job_id: str = None
):
    """
    Function to train YOLOv11 model
    """
    # Set up logging
    if job_id is None:
        job_id = f"training_job_{int(time.time())}"
    
    with LogCapture(job_id):
        try:
            print(f"Starting training job {job_id}")
            print(f"Dataset ID: {dataset_id}")
            print(f"Training parameters: epochs={epochs}, batch_size={batch_size}, image_size={image_size}, learning_rate={learning_rate}")
            
            # Get dataset information
            dataset = EyeRecognitionSampleDAO.get_data_train_by_id(dataset_id)
            if not dataset:
                print(f"Error: Dataset with ID {dataset_id} not found")
                return
            
            dataset_path = dataset['dataTrainPath']
            yaml_path = os.path.join(dataset_path, 'data.yaml')
            
            if not os.path.exists(yaml_path):
                print(f"Error: data.yaml not found in dataset directory")
                return
            
            # Create training history record
            train_history = TrainDetectionHistory(
                epochs=epochs,
                batchSize=batch_size,
                imageSize=image_size,
                learningRate=learning_rate,
                tblDetectEyeDataTrainId=dataset_id,
                tblEyeDetectionModelId=None  # Will set this after training is complete
            )
            
            history_id = EyeRecognitionSampleHistoryDAO.create_train_history(train_history)
            print(f"Created training history record with ID {history_id}")
            
            # Set up YOLOv11 training
            print("Initializing YOLOv11 model...")
            
            # If using pretrained model
            pretrained_weights = None
            if pretrained_model_id:
                model_info = EyeRecognitionModelDAO.get_model_by_id(pretrained_model_id)
                if model_info:
                    pretrained_weights = model_info['modelLink']
                    print(f"Using pretrained model: {model_info['modelName']}")
            
            # Start training
            print("Starting YOLOv11 training...")
            
            # Here we'd normally call the YOLOv11 training code
            # For demonstration, we're using a mock training process:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            
            # Mock some training output
            for epoch in range(epochs):
                time.sleep(1)  # Simulate training time
                print(f"Epoch {epoch+1}/{epochs}, loss: {5.0 - 4.0 * (epoch+1) / epochs:.4f}, mAP: {0.3 + 0.6 * (epoch+1) / epochs:.4f}")
            
            # Save the model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"eye_detector_yolov11_{timestamp}"
            model_filename = f"{model_name}.pt"
            model_path = os.path.join(MODEL_DIR, model_filename)
            
            # In a real implementation, we'd save the actual trained model
            # model.save(model_path)
            # For demonstration, we'll just copy a pretrained model or create an empty file
            if pretrained_weights and os.path.exists(pretrained_weights):
                shutil.copy(pretrained_weights, model_path)
            else:
                # Create empty model file for demonstration
                with open(model_path, 'w') as f:
                    f.write("Mock YOLOv11 model")
            
            print(f"Model saved to {model_path}")
            
            # Create model record
            map_metric = 0.85  # Mock mAP metric
            model_record = EyeRecognitionModel(
                modelName=model_name,
                mapMetric=map_metric,
                createDate=datetime.now(),
                isActive=0,  # Not active by default
                modelLink=model_path
            )
            
            model_id = EyeRecognitionModelDAO.create_model(model_record)
            print(f"Created model record with ID {model_id}")
            
            # Update training history with model ID
            train_history.tblEyeDetectionModelId = model_id
            EyeRecognitionSampleHistoryDAO.update_train_history(history_id, train_history)
            
            print(f"Training job {job_id} completed successfully")
            
        except Exception as e:
            print(f"Error in training job: {str(e)}")
            # Log the error but don't re-raise it so the thread can complete


@router.get("/history")
async def get_training_history():
    """
    Get all training history
    """
    history = EyeRecognitionSampleHistoryDAO.get_all_train_history()
    return {"training_history": history}


@router.get("/history/{history_id}")
async def get_training_history_by_id(history_id: int):
    """
    Get training history by ID
    """
    history = EyeRecognitionSampleHistoryDAO.get_train_history_by_id(history_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Training history with ID {history_id} not found")
    return {"training_history": history}


@router.post("/start")
async def start_training(
    background_tasks: BackgroundTasks,
    dataset_id: int = Form(...),
    epochs: int = Form(...),
    batch_size: int = Form(...),
    image_size: int = Form(...),
    learning_rate: float = Form(...),
    pretrained_model_id: Optional[int] = Form(None)
):
    """
    Start a new training job
    """
    # Validate dataset
    dataset = EyeRecognitionSampleDAO.get_data_train_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    # Validate pretrained model if provided
    if pretrained_model_id:
        model = EyeRecognitionModelDAO.get_model_by_id(pretrained_model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Pretrained model with ID {pretrained_model_id} not found")
    
    # Generate a job ID
    job_id = f"training_job_{int(time.time())}"
    
    # Start training in background
    background_tasks.add_task(
        train_yolov11_model,
        dataset_id,
        epochs,
        batch_size,
        image_size,
        learning_rate,
        pretrained_model_id,
        job_id
    )
    
    return {
        "message": "Training started",
        "job_id": job_id
    }


@router.get("/logs/{job_id}")
async def get_training_logs(job_id: str):
    """
    Get logs for a training job
    """
    if job_id not in training_logs:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")
    
    return {"job_id": job_id, "logs": training_logs[job_id]}


@router.post("/activate-model")
async def activate_trained_model(history_id: int = Form(...)):
    """
    Activate a model that was trained in a training job
    """
    history = EyeRecognitionSampleHistoryDAO.get_train_history_by_id(history_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Training history with ID {history_id} not found")
    
    model_id = history.get('tblEyeDetectionModelId')
    if not model_id:
        raise HTTPException(status_code=400, detail=f"No model associated with training history {history_id}")
    
    success = EyeRecognitionModelDAO.set_active_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to activate model")
    
    return {"message": f"Model from training history {history_id} activated successfully"}


@router.delete("/history/{history_id}")
async def delete_training_history(history_id: int):
    """
    Delete a training history entry
    """
    history = EyeRecognitionSampleHistoryDAO.get_train_history_by_id(history_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Training history with ID {history_id} not found")
    
    success = EyeRecognitionSampleHistoryDAO.delete_train_history(history_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete training history")
    
    return {"message": f"Training history with ID {history_id} deleted successfully"}