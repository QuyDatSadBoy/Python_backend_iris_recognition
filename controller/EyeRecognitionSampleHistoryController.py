import os
import json
import yaml
import torch
import shutil
import glob
from fastapi import APIRouter, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
from io import StringIO
import sys

from dao.EyeRecognitionModelDAO import EyeRecognitionModelDAO
from dao.EyeRecognitionSampleDAO import EyeRecognitionSampleDAO
from dao.EyeRecognitionSampleHistoryDAO import EyeRecognitionSampleHistoryDAO
from entity.EyeRecognitionSampleHistory import TrainDetectionHistory
from entity.EyeRecognitionModel import EyeRecognitionModel
from ultralytics import YOLO
from db_connection import get_connection

# Define the router
router = APIRouter(prefix="/training", tags=["Training"])

# Path constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
UPLOADS_DIR = "/home/quydat09/iris_rcog/eye-recognition-system/uploads"

# Make sure the directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

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
    job_id: str = None
):
    """
    Function to train YOLOv11 model using model with ID 1 as pretrained
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
            yaml_path = dataset['detailFilePath']
            
            print(f"Dataset path: {dataset_path}")
            print(f"Using YAML file: {yaml_path}")
            
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
            
            # Always use model ID 1 as pretrained
            model_info = EyeRecognitionModelDAO.get_model_by_id(1)
            if model_info:
                pretrained_weights = os.path.join(BASE_DIR, model_info['modelLink'])
                print(f"Using model ID 1 as pretrained model: {model_info['modelName']}")
            else:
                # Fallback to YOLO default if model 1 not found
                pretrained_weights = 'yolov8n.pt'
                print(f"Model ID 1 not found. Using default YOLOv8n")
            
            # Start training
            try:
                # Load model
                print(f"Loading model from {pretrained_weights}")
                model = YOLO(pretrained_weights)
                
                # Train model
                print(f"Starting training with YAML: {yaml_path}")
                results = model.train(
                    data=yaml_path,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=image_size,
                    lr0=learning_rate,
                    device='0' if torch.cuda.is_available() else 'cpu'
                )
                
                # Lấy đường dẫn đến thư mục lưu kết quả
                save_dir = getattr(results, 'save_dir', None)
                if save_dir:
                    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
                    last_model_path = os.path.join(save_dir, 'weights', 'last.pt')
                else:
                    # Fallback nếu không tìm thấy save_dir
                    run_dirs = glob.glob(os.path.join('runs', 'detect', 'train*'))
                    if run_dirs:
                        latest_run = max(run_dirs, key=os.path.getmtime)
                        best_model_path = os.path.join(latest_run, 'weights', 'best.pt')
                        last_model_path = os.path.join(latest_run, 'weights', 'last.pt')
                    else:
                        raise Exception("Could not find model weights directory")

                # Kiểm tra và sử dụng best.pt nếu có, không thì dùng last.pt
                if os.path.exists(best_model_path):
                    chosen_model_path = best_model_path
                    print(f"Using best model from: {best_model_path}")
                elif os.path.exists(last_model_path):
                    chosen_model_path = last_model_path
                    print(f"Best model not found, using last model from: {last_model_path}")
                else:
                    raise Exception("Neither best nor last model found")

                # Save the model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"eye_detector_yolo_{timestamp}"
                model_filename = f"{model_name}.pt"
                model_path = os.path.join(MODEL_DIR, model_filename)

                shutil.copy(chosen_model_path, model_path)
                print(f"Model copied from {chosen_model_path} to {model_path}")
                
                # Calculate relative path for database
                model_rel_path = os.path.relpath(model_path, BASE_DIR)
                print(f"Model saved to {model_path} (rel: {model_rel_path})")
                
                # Get mAP metric
                metrics = results.results_dict
                map_metric = metrics.get('metrics/mAP50-95(B)', 0.0)
                print(f"Model mAP: {map_metric}")
                
                # Create model record
                model_record = EyeRecognitionModel(
                    modelName=model_name,
                    mapMetric=map_metric,
                    createDate=datetime.now(),
                    isActive=0,  # Not active by default
                    modelLink=model_rel_path
                )
                
                model_id = EyeRecognitionModelDAO.create_model(model_record)
                print(f"Created model record with ID {model_id}")
                
                # Update training history with model ID
                train_history.tblEyeDetectionModelId = model_id
                EyeRecognitionSampleHistoryDAO.update_train_history(history_id, train_history)
                
            except Exception as train_error:
                print(f"Error during training: {str(train_error)}")
                
                # Create a fallback model if training fails
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"eye_detector_fallback_{timestamp}"
                model_filename = f"{model_name}.pt"
                model_path = os.path.join(MODEL_DIR, model_filename)
                
                # Use pretrained model as fallback
                if os.path.exists(pretrained_weights):
                    shutil.copy(pretrained_weights, model_path)
                    print(f"Using pretrained model as fallback")
                else:
                    # Create empty file as placeholder
                    with open(model_path, 'w') as f:
                        f.write("Fallback model - Training failed")
                    print(f"Created empty fallback model")
                
                model_rel_path = os.path.relpath(model_path, BASE_DIR)
                
                # Create model record for fallback
                model_record = EyeRecognitionModel(
                    modelName=model_name + " (Training Failed)",
                    mapMetric=0.0,
                    createDate=datetime.now(),
                    isActive=0,  # Not active
                    modelLink=model_rel_path
                )
                
                model_id = EyeRecognitionModelDAO.create_model(model_record)
                print(f"Created fallback model record with ID {model_id}")
                
                # Update training history with model ID
                train_history.tblEyeDetectionModelId = model_id
                EyeRecognitionSampleHistoryDAO.update_train_history(history_id, train_history)
            
            print(f"Training job {job_id} completed")
            
        except Exception as e:
            print(f"Unhandled error in training job: {str(e)}")
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
    dataset_id: int = Form(...),
    epochs: int = Form(...),
    batch_size: int = Form(...),
    image_size: int = Form(...),
    learning_rate: float = Form(...)
):
    """
    Start a new training job using model ID 1 as pretrained and wait for it to complete
    """
    # Validate dataset
    dataset = EyeRecognitionSampleDAO.get_data_train_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    # Kiểm tra xem file YAML có tồn tại không
    yaml_path = dataset.get('detailFilePath')
    if not yaml_path or not os.path.exists(yaml_path):
        raise HTTPException(status_code=400, detail=f"YAML file not found at {yaml_path}")
    
    # Generate a job ID
    job_id = f"training_job_{int(time.time())}"
    
    # Thực hiện training ngay lập tức (không sử dụng background_tasks)
    job_logs = StringIO()
    sys.stdout = job_logs
    
    try:
        print(f"Starting training job {job_id}")
        print(f"Dataset ID: {dataset_id}")
        print(f"Training parameters: epochs={epochs}, batch_size={batch_size}, image_size={image_size}, learning_rate={learning_rate}")
        
        dataset_path = dataset['dataTrainPath']
        print(f"Dataset path: {dataset_path}")
        print(f"Using YAML file: {yaml_path}")
        
        # Kiểm tra và điều chỉnh file YAML
        try:
            with open(yaml_path, 'r') as file:
                yaml_content = yaml.safe_load(file)
            
            # Tìm thư mục thực tế chứa train/valid
            data_subdir = None
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    # Kiểm tra xem thư mục này có chứa train/valid không
                    if os.path.exists(os.path.join(item_path, 'train')) or os.path.exists(os.path.join(item_path, 'valid')):
                        data_subdir = item
                        break
            
            # Nếu tìm thấy thư mục con chứa dữ liệu, điều chỉnh đường dẫn
            if data_subdir:
                print(f"Found data subdirectory: {data_subdir}")
                new_path = os.path.join(dataset_path, data_subdir)
                yaml_content['path'] = new_path
                
                # Kiểm tra xem train/valid có nằm trong thư mục hiện tại không
                if 'train' in yaml_content and yaml_content['train'].startswith('../'):
                    yaml_content['train'] = yaml_content['train'].replace('../', './')
                if 'val' in yaml_content and yaml_content['val'].startswith('../'):
                    yaml_content['val'] = yaml_content['val'].replace('../', './')
                
                # Lưu file YAML đã điều chỉnh
                with open(yaml_path, 'w') as file:
                    yaml.dump(yaml_content, file)
                
                print(f"Updated YAML file with correct paths: {yaml_content}")
        except Exception as e:
            print(f"Error adjusting YAML file: {str(e)}")
        
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
        
        # Always use model ID 1 as pretrained
        model_info = EyeRecognitionModelDAO.get_model_by_id(1)
        if model_info:
            pretrained_weights = os.path.join(BASE_DIR, model_info['modelLink'])
            print(f"Using model ID 1 as pretrained model: {model_info['modelName']}")
        else:
            # Fallback to YOLO default if model 1 not found
            pretrained_weights = 'yolov8n.pt'
            print(f"Model ID 1 not found. Using default YOLOv8n")
        
        # Start training
        model_record = None
        model_id = None
        
        try:
            # Load model
            print(f"Loading model from {pretrained_weights}")
            model = YOLO(pretrained_weights)
            
            # Train model
            print(f"Starting training with YAML: {yaml_path}")
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                lr0=learning_rate,
                device='0' if torch.cuda.is_available() else 'cpu'
            )
            
            # Lấy đường dẫn đến thư mục lưu kết quả
            save_dir = getattr(results, 'save_dir', None)
            if save_dir:
                best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
                last_model_path = os.path.join(save_dir, 'weights', 'last.pt')
            else:
                # Fallback nếu không tìm thấy save_dir
                run_dirs = glob.glob(os.path.join('runs', 'detect', 'train*'))
                if run_dirs:
                    latest_run = max(run_dirs, key=os.path.getmtime)
                    best_model_path = os.path.join(latest_run, 'weights', 'best.pt')
                    last_model_path = os.path.join(latest_run, 'weights', 'last.pt')
                else:
                    raise Exception("Could not find model weights directory")

            # Kiểm tra và sử dụng best.pt nếu có, không thì dùng last.pt
            if os.path.exists(best_model_path):
                chosen_model_path = best_model_path
                print(f"Using best model from: {best_model_path}")
            elif os.path.exists(last_model_path):
                chosen_model_path = last_model_path
                print(f"Best model not found, using last model from: {last_model_path}")
            else:
                raise Exception("Neither best nor last model found")

            # Save the model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"eye_detector_yolo_{timestamp}"
            model_filename = f"{model_name}.pt"
            model_path = os.path.join(MODEL_DIR, model_filename)

            shutil.copy(chosen_model_path, model_path)
            print(f"Model copied from {chosen_model_path} to {model_path}")
            
            # Calculate relative path for database
            model_rel_path = os.path.relpath(model_path, BASE_DIR)
            print(f"Model saved to {model_path} (rel: {model_rel_path})")
            
            # Get mAP metric from results
            metrics = results.results_dict
            map_metric = metrics.get('metrics/mAP50-95(B)', 0.0)
            print(f"Model mAP: {map_metric}")
            
            # Create model record
            model_record = EyeRecognitionModel(
                modelName=model_name,
                mapMetric=map_metric,
                createDate=datetime.now(),
                isActive=0,  # Not active by default
                modelLink=model_rel_path
            )
            
            model_id = EyeRecognitionModelDAO.create_model(model_record)
            print(f"Created model record with ID {model_id}")
            
            # Update training history with model ID
            train_history.tblEyeDetectionModelId = model_id
            EyeRecognitionSampleHistoryDAO.update_train_history(history_id, train_history)
            
        except Exception as train_error:
            print(f"Error during training: {str(train_error)}")
            
            # Create a fallback model if training fails
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"eye_detector_fallback_{timestamp}"
            model_filename = f"{model_name}.pt"
            model_path = os.path.join(MODEL_DIR, model_filename)
            
            # Use pretrained model as fallback
            if os.path.exists(pretrained_weights):
                shutil.copy(pretrained_weights, model_path)
                print(f"Using pretrained model as fallback")
            else:
                # Create empty file as placeholder
                with open(model_path, 'w') as f:
                    f.write("Fallback model - Training failed")
                print(f"Created empty fallback model")
            
            model_rel_path = os.path.relpath(model_path, BASE_DIR)
            
            # Create model record for fallback
            model_record = EyeRecognitionModel(
                modelName=model_name + " (Training Failed)",
                mapMetric=0.0,
                createDate=datetime.now(),
                isActive=0,  # Not active
                modelLink=model_rel_path
            )
            
            model_id = EyeRecognitionModelDAO.create_model(model_record)
            print(f"Created fallback model record with ID {model_id}")
            
            # Update training history with model ID
            train_history.tblEyeDetectionModelId = model_id
            EyeRecognitionSampleHistoryDAO.update_train_history(history_id, train_history)
        
        print(f"Training job {job_id} completed")
        
        # Lưu logs cho tham chiếu sau này
        training_logs[job_id] = job_logs.getvalue()
        
        # Khôi phục stdout
        sys.stdout = sys.__stdout__
        
        # Lấy thông tin model đã train để trả về
        if model_id:
            model_details = EyeRecognitionModelDAO.get_model_by_id(model_id)
        else:
            model_details = None
        
        # Trả về thông tin
        return {
            "message": "Training completed",
            "job_id": job_id,
            "history_id": history_id,
            "logs": training_logs[job_id],
            "model": model_details
        }
        
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")


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