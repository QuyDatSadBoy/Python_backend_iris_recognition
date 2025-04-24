import os
import shutil
import zipfile
import yaml
import json
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import glob
from datetime import datetime

from dao.EyeRecognitionSampleDAO import EyeRecognitionSampleDAO
from entity.EyeRecognitionSample import DetectEyeDataTrain, DetectEyeData

# Define the router
router = APIRouter(prefix="/samples", tags=["Samples"])

# Path constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, "../eye-recognition-system/uploads")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

# Make sure the directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


class DatasetMetadata(BaseModel):
    name: str
    description: Optional[str] = None
    classes: List[str]
    total_images: int
    train_images: int
    val_images: int
    test_images: Optional[int] = None


def process_yolo_dataset(dataset_path, dataset_id):
    """
    Process a YOLO dataset and save information to the database
    """
    # Expected YOLO dataset structure:
    # dataset/
    # ├── data.yaml
    # ├── train/
    # │   ├── images/
    # │   └── labels/
    # ├── val/
    # │   ├── images/
    # │   └── labels/
    # └── test/ (optional)
    #     ├── images/
    #     └── labels/
    
    # Read the dataset configuration
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(yaml_path):
        raise HTTPException(status_code=400, detail="Invalid YOLO dataset. Missing data.yaml file.")
    
    try:
        with open(yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse data.yaml: {str(e)}")
    
    # Process and save each image and label to the database
    eye_data_list = []
    
    # Process train images
    train_images_dir = os.path.join(dataset_path, 'train', 'images')
    train_labels_dir = os.path.join(dataset_path, 'train', 'labels')
    
    if os.path.exists(train_images_dir) and os.path.exists(train_labels_dir):
        for img_file in glob.glob(os.path.join(train_images_dir, '*.*')):
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            label_file = os.path.join(train_labels_dir, base_name + '.txt')
            
            if os.path.exists(label_file):
                # Create a relative path for image and label
                rel_img_path = os.path.relpath(img_file, start=BASE_DIR)
                rel_label_path = os.path.relpath(label_file, start=BASE_DIR)
                
                eye_data = DetectEyeData(
                    imageLink=rel_img_path,
                    labelLink=rel_label_path,
                    tblDetectEyeDataTrainId=dataset_id
                )
                eye_data_list.append(eye_data)
    
    # Process validation images
    val_images_dir = os.path.join(dataset_path, 'val', 'images')
    val_labels_dir = os.path.join(dataset_path, 'val', 'labels')
    
    if os.path.exists(val_images_dir) and os.path.exists(val_labels_dir):
        for img_file in glob.glob(os.path.join(val_images_dir, '*.*')):
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            label_file = os.path.join(val_labels_dir, base_name + '.txt')
            
            if os.path.exists(label_file):
                # Create a relative path for image and label
                rel_img_path = os.path.relpath(img_file, start=BASE_DIR)
                rel_label_path = os.path.relpath(label_file, start=BASE_DIR)
                
                eye_data = DetectEyeData(
                    imageLink=rel_img_path,
                    labelLink=rel_label_path,
                    tblDetectEyeDataTrainId=dataset_id
                )
                eye_data_list.append(eye_data)
    
    # Save all eye data entries to the database
    if eye_data_list:
        rows_affected = EyeRecognitionSampleDAO.create_multiple_eye_data(eye_data_list)
        return rows_affected
    
    return 0


@router.get("/datasets")
async def get_all_datasets():
    """
    Get all datasets
    """
    datasets = EyeRecognitionSampleDAO.get_all_data_trains()
    return {"datasets": datasets}


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: int):
    """
    Get a dataset by ID
    """
    dataset = EyeRecognitionSampleDAO.get_data_train_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    return {"dataset": dataset}


@router.get("/datasets/{dataset_id}/samples")
async def get_dataset_samples(dataset_id: int):
    """
    Get all eye samples for a dataset
    """
    dataset = EyeRecognitionSampleDAO.get_data_train_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    samples = EyeRecognitionSampleDAO.get_all_eye_data(dataset_id)
    return {"samples": samples}


@router.post("/datasets/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    dataset_zip: UploadFile = File(...),
    dataset_name: str = Form(...),
    description: Optional[str] = Form(None)
):
    """
    Upload a YOLO dataset in ZIP format
    """
    # Check if it's a ZIP file
    if not dataset_zip.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    # Create a directory for this dataset
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save the ZIP file temporarily
    zip_path = os.path.join(dataset_dir, "dataset.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(dataset_zip.file, buffer)
    
    # Create a detail file with metadata
    detail_path = os.path.join(dataset_dir, "metadata.json")
    metadata = {
        "name": dataset_name,
        "description": description,
        "original_filename": dataset_zip.filename,
        "upload_time": str(datetime.now())
    }
    
    with open(detail_path, "w") as f:
        json.dump(metadata, f)
    
    # Extract the ZIP file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
    except Exception as e:
        # Clean up on extraction failure
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to extract ZIP file: {str(e)}")
    
    # Create a data train entry in the database
    data_train = DetectEyeDataTrain(
        dataTrainPath=dataset_dir,
        detailFilePath=detail_path
    )
    
    dataset_id = EyeRecognitionSampleDAO.create_data_train(data_train)
    
    # Process the dataset in the background
    background_tasks.add_task(process_yolo_dataset, dataset_dir, dataset_id)
    
    return {
        "message": "Dataset uploaded successfully. Processing in progress.",
        "dataset_id": dataset_id
    }


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int):
    """
    Delete a dataset
    """
    dataset = EyeRecognitionSampleDAO.get_data_train_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found")
    
    # Delete the dataset from the database
    success = EyeRecognitionSampleDAO.delete_data_train(dataset_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete dataset")
    
    # Delete the dataset directory if it exists
    dataset_dir = dataset['dataTrainPath']
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir, ignore_errors=True)
    
    return {"message": f"Dataset with ID {dataset_id} deleted successfully"}


@router.delete("/samples/{sample_id}")
async def delete_eye_sample(sample_id: int):
    """
    Delete an eye sample
    """
    sample = EyeRecognitionSampleDAO.get_eye_data_by_id(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail=f"Sample with ID {sample_id} not found")
    
    # Delete the sample from the database
    success = EyeRecognitionSampleDAO.delete_eye_data(sample_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete sample")
    
    return {"message": f"Sample with ID {sample_id} deleted successfully"}