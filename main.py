import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from controller.EyeRecognitionModelController import router as model_router
from controller.EyeRecognitionSampleController import router as sample_router
from controller.EyeRecognitionSampleHistoryController import router as history_router

# Load environment variables
load_dotenv()

# Create the FastAPI application
app = FastAPI(
    title="Eye Recognition Training Service",
    description="API for training and managing eye recognition models",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(model_router)
app.include_router(sample_router)
app.include_router(history_router)

# Path constants from environment variables or defaults
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, os.getenv("UPLOADS_DIR", "../eye-recognition-system/uploads"))
MODEL_DIR = os.path.join(BASE_DIR, os.getenv("MODEL_DIR", "models"))

# Make sure the necessary directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "eyes"), exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "faces"), exist_ok=True)

# Mount the uploads directory for static file access
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


@app.get("/")
async def root():
    """Root endpoint that returns basic service information"""
    return {
        "service": "Eye Recognition Training Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run("main:app", host=host, port=port, reload=True)