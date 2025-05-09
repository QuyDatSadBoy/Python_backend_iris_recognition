from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from controller.EyeDetectionModelController import router as eye_detection_model_router
from controller.DetectEyeDataTrainController import router as detect_eye_data_train_router
from controller.DetectEyeDataController import router as detect_eye_data_router
from controller.TrainDetectionHistoryController import router as train_detection_history_router

app = FastAPI(
    title="Eye Recognition Training Service",
    description="API for training and managing eye recognition models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(eye_detection_model_router)
app.include_router(detect_eye_data_train_router)
app.include_router(detect_eye_data_router)
app.include_router(train_detection_history_router)

# Static files
UPLOADS_DIR = "/home/quydat09/iris_rcog/eye-recognition-system/uploads"
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

@app.get("/")
async def root():
    return {
        "service": "Eye Recognition Training Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run("main:app", host=host, port=port, reload=True)