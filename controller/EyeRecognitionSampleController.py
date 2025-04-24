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
import xml.etree.ElementTree as ET

from dao.EyeRecognitionSampleDAO import EyeRecognitionSampleDAO
from entity.EyeRecognitionSample import DetectEyeDataTrain, DetectEyeData
from db_connection import get_connection



# Define the router
router = APIRouter(prefix="/samples", tags=["Samples"])

# Path constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = "/home/quydat09/iris_rcog/eye-recognition-system/uploads"
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

# Make sure the directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "eyes"), exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "faces"), exist_ok=True)


class DatasetMetadata(BaseModel):
    name: str
    description: Optional[str] = None
    classes: List[str]
    total_images: int
    train_images: int
    val_images: int
    test_images: Optional[int] = None


def find_yaml_file(directory):
    """
    Tìm file yaml đầu tiên trong thư mục hoặc thư mục con
    """
    # Tìm trong thư mục hiện tại
    yaml_files = glob.glob(os.path.join(directory, "*.yaml")) + glob.glob(os.path.join(directory, "*.yml"))
    if yaml_files:
        return yaml_files[0]
    
    # Tìm trong các thư mục con (chỉ đi sâu 1 cấp)
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            yaml_files = glob.glob(os.path.join(subdir_path, "*.yaml")) + glob.glob(os.path.join(subdir_path, "*.yml"))
            if yaml_files:
                return yaml_files[0]
            
            # Kiểm tra thư mục data.yaml cụ thể
            data_yaml = os.path.join(subdir_path, "data.yaml")
            if os.path.exists(data_yaml):
                return data_yaml
    
    return None


def is_image_file(filename):
    """
    Kiểm tra xem file có phải là ảnh hay không
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    ext = os.path.splitext(filename)[1].lower()
    return ext in image_extensions


def is_xml_label_file(filename):
    """
    Kiểm tra xem file có phải là file XML nhãn hay không
    """
    return filename.lower().endswith('.xml')


def process_yolo_dataset(dataset_path, dataset_id):
    """
    Xử lý bộ dữ liệu YOLO và lưu thông tin vào cơ sở dữ liệu
    """
    print(f"Đang xử lý bộ dữ liệu tại: {dataset_path}")
    
    # Tìm file yaml
    yaml_file = find_yaml_file(dataset_path)
    if not yaml_file:
        print(f"Không tìm thấy file yaml trong thư mục {dataset_path}")
        # Tạo file yaml mặc định
        yaml_file = os.path.join(dataset_path, "data.yaml")
        with open(yaml_file, 'w') as f:
            yaml_content = {
                'path': dataset_path,
                'train': 'train/images',
                'val': 'valid/images',
                'names': {
                    0: 'iris',
                    1: 'pupil',
                    2: 'eye'
                }
            }
            yaml.dump(yaml_content, f)
        print(f"Đã tạo file yaml mặc định: {yaml_file}")
    
    print(f"Tìm thấy file yaml: {yaml_file}")
    
    # Cập nhật đường dẫn file yaml trong cơ sở dữ liệu
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            sql = "UPDATE tblDetectEyeDataTrain SET detailFilePath = %s WHERE id = %s"
            cursor.execute(sql, (yaml_file, dataset_id))
            connection.commit()
    except Exception as e:
        print(f"Lỗi khi cập nhật đường dẫn file yaml: {str(e)}")
    finally:
        connection.close()
    
    
    # Tìm tất cả các ảnh và file label trong thư mục
    eye_data_list = []
    image_path_file = []
    label_path_file = []
    image_count = 0
    label_count = 0
    
    # Tìm tất cả các thư mục trong dataset
    for root, dirs, files in os.walk(dataset_path):
        # Kiểm tra mỗi file trong thư mục
        image_path_file += [{ 'path':os.path.join(root, f.strip()),'name':f.strip()[:-4]} for f in files if f.endswith('.jpg') or f.endswith('.png')]
        label_path_file += [{'path':os.path.join(root, f.strip()),'name':f.strip()[:-4]} for f in files if f.endswith('.xml') or f.endswith('.txt')]
        
    
    for img_path in image_path_file:
        for label_path in label_path_file:
            if img_path['name'] == label_path['name']:
                eye_data = DetectEyeData(
                    imageLink=img_path['path'],
                    labelLink=label_path['path'],
                    tblDetectEyeDataTrainId=dataset_id
                )
                eye_data_list.append(eye_data)
    
    print(f"Tổng số ảnh: {len(eye_data_list)}, Tổng số file nhãn: {len(eye_data_list)}")
    
    
    # Lưu tất cả các dữ liệu vào cơ sở dữ liệu
    if eye_data_list:
        try:
            rows_affected = EyeRecognitionSampleDAO.create_multiple_eye_data(eye_data_list)
            print(f"Đã lưu {rows_affected} dòng dữ liệu vào cơ sở dữ liệu")
            return rows_affected
        except Exception as e:
            print(f"Lỗi khi lưu dữ liệu vào cơ sở dữ liệu: {str(e)}")
            # In thông tin chi tiết về dữ liệu đang cố gắng lưu
            if len(eye_data_list) > 0:
                print(f"Mẫu dữ liệu đầu tiên: {eye_data_list[0]}")
            return 0
    
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
    
    # Xóa thư mục cũ nếu tồn tại
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save the ZIP file temporarily
    zip_path = os.path.join(dataset_dir, "dataset.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(dataset_zip.file, buffer)
    
    # Extract the ZIP file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print(f"Đã giải nén file ZIP vào: {dataset_dir}")
    except Exception as e:
        # Clean up on extraction failure
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to extract ZIP file: {str(e)}")
    
    # Tìm file yaml
    yaml_file = find_yaml_file(dataset_dir)
    if not yaml_file:
        # Nếu không tìm thấy, tạo file data.yaml với nội dung mặc định
        yaml_file = os.path.join(dataset_dir, "data.yaml")
        with open(yaml_file, 'w') as f:
            yaml_content = {
                'path': dataset_dir,
                'train': 'train/images',
                'val': 'valid/images',
                'names': {
                    0: 'iris',
                    1: 'pupil',
                    2: 'eye'
                }
            }
            yaml.dump(yaml_content, f)
        print(f"Đã tạo file yaml mặc định: {yaml_file}")
    
    # Create a data train entry in the database
    data_train = DetectEyeDataTrain(
        dataTrainPath=dataset_dir,
        detailFilePath=yaml_file  # Lưu đường dẫn file yaml
    )
    
    dataset_id = EyeRecognitionSampleDAO.create_data_train(data_train)
    print(f"Đã tạo bản ghi dataset với ID: {dataset_id}")
    
    # Xử lý dataset ngay lập tức thay vì sử dụng background task
    try:
        rows_affected = process_yolo_dataset(dataset_dir, dataset_id)
        print(f"Đã xử lý dataset và lưu {rows_affected} dòng dữ liệu")
    except Exception as e:
        print(f"Lỗi khi xử lý dataset: {str(e)}")
        # Vẫn trả về kết quả thành công, nhưng ghi log lỗi
    
    return {
        "message": "Dataset uploaded successfully. Processing completed.",
        "dataset_id": dataset_id,
        "dataset_path": dataset_dir,
        "yaml_file": yaml_file,
        "samples_processed": rows_affected if 'rows_affected' in locals() else 0
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