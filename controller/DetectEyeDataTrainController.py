from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import Optional, List
from dao.DetectEyeDataTrainDAO import DetectEyeDataTrainDAO
from dao.DetectEyeDataDAO import DetectEyeDataDAO
from entity.DetectEyeDataTrain import DetectEyeDataTrain
from entity.DetectEyeData import DetectEyeData
import os
import shutil
import zipfile
import yaml

router = APIRouter(prefix="/training-data", tags=["DetectEyeDataTrain"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

@router.get("/")
async def getAllDetectEyeDataTrain() -> List[DetectEyeDataTrain]:
    """GET /training-data - Lấy tất cả training data"""
    return DetectEyeDataTrainDAO.getAllDetectEyeDataTrain()

@router.get("/{id}")
async def getDetectEyeDataTrainById(id: int) -> DetectEyeDataTrain:
    """GET /training-data/{id} - Lấy training data theo ID"""
    data = DetectEyeDataTrainDAO.getDetectEyeDataTrainById(id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Training data with ID {id} not found")
    return data

@router.post("/")
async def createDetectEyeDataTrain(
    datasetName: str = Form(...),
    datasetZip: UploadFile = File(...),
    description: Optional[str] = Form(None)
) -> DetectEyeDataTrain:
    """POST /training-data - Tạo training data mới"""
    # Tạo thư mục cho dataset
    dataset_dir = os.path.join(DATASET_DIR, datasetName)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Lưu và giải nén file
    zip_path = os.path.join(dataset_dir, "dataset.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(datasetZip.file, buffer)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    # Tìm file yaml
    yaml_file = None
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                yaml_file = os.path.join(root, file)
                break
        if yaml_file:
            break
    
    if not yaml_file:
        # Tạo file yaml mặc định
        yaml_file = os.path.join(dataset_dir, "data.yaml")
        with open(yaml_file, 'w') as f:
            yaml_content = {
                'path': dataset_dir,
                'train': 'train/images',
                'val': 'valid/images',
                'names': {0: 'iris', 1: 'pupil', 2: 'eye'}
            }
            yaml.dump(yaml_content, f)
    
    # Tạo training data
    training_data = DetectEyeDataTrain(
        dataTrainPath=dataset_dir,
        detailFilePath=yaml_file
    )
    
    data_id = DetectEyeDataTrainDAO.createDetectEyeDataTrain(training_data)
    
    # Xử lý và tạo detect eye data
    eye_data_list = []
    image_files = []
    label_files = []
    
    # Tìm tất cả các file ảnh và label
    for root, dirs, files in os.walk(dataset_dir):
        image_files += [{'path': os.path.join(root, f.strip()), 'name': f.strip()[:-4]} 
                       for f in files if f.endswith('.jpg') or f.endswith('.png')]
        label_files += [{'path': os.path.join(root, f.strip()), 'name': f.strip()[:-4]} 
                       for f in files if f.endswith('.xml') or f.endswith('.txt')]
    
    # Ghép cặp ảnh và label
    for img_file in image_files:
        for label_file in label_files:
            if img_file['name'] == label_file['name']:
                eye_data = DetectEyeData(
                    imageLink=img_file['path'],
                    labelLink=label_file['path'],
                    tblDetectEyeDataTrainId=data_id
                )
                eye_data_list.append(eye_data)
    
    # Lưu vào database
    if eye_data_list:
        DetectEyeDataDAO.createDetectEyeDataList(eye_data_list)
    
    training_data.id = data_id
    return training_data

@router.delete("/{id}")
async def deleteDetectEyeDataTrain(id: int) -> bool:
    """DELETE /training-data/{id} - Xóa training data"""
    success = DetectEyeDataTrainDAO.deleteDetectEyeDataTrain(id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Training data with ID {id} not found")
    return success