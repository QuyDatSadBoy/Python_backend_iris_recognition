# Dịch Vụ Nhận Dạng Mắt (Eye Recognition Service)

## Tổng Quan

Dịch vụ Nhận Dạng Mắt là một API REST được xây dựng bằng FastAPI, được thiết kế để huấn luyện và quản lý các mô hình nhận dạng mắt sử dụng công nghệ YOLO (You Only Look Once) và OpenCV. Dự án này cung cấp các chức năng toàn diện cho việc huấn luyện, đánh giá và triển khai các mô hình phát hiện mắt.

## Tính Năng Chính

### 🔍 Quản Lý Mô Hình
- Tải lên và quản lý các mô hình YOLO
- Theo dõi hiệu suất và metrics của mô hình
- Lưu trữ thông tin cấu hình mô hình

### 📊 Quản Lý Dữ Liệu Huấn Luyện
- Upload và xử lý dữ liệu huấn luyện
- Quản lý datasets cho việc huấn luyện mô hình
- Xem trước và kiểm tra chất lượng dữ liệu

### 🚀 Huấn Luyện Mô Hình
- Huấn luyện mô hình YOLO với dữ liệu tùy chỉnh
- Theo dõi tiến trình huấn luyện real-time
- Lưu trữ lịch sử huấn luyện và kết quả

### 📈 Theo Dõi và Báo Cáo
- Lịch sử huấn luyện chi tiết
- Metrics và đánh giá hiệu suất
- Trực quan hóa kết quả huấn luyện

## Công Nghệ Sử Dụng

- **Framework**: FastAPI 0.110.0
- **Server**: Uvicorn 0.29.0
- **Machine Learning**: 
  - Ultralytics YOLO 8.1.31
  - PyTorch 2.2.1
  - TorchVision 0.17.1
- **Computer Vision**: OpenCV 4.9.0.80
- **Database**: MySQL (PyMySQL 1.1.0)
- **Khác**: NumPy, Pillow, Pydantic

## Cài Đặt

### Yêu Cầu Hệ Thống
- Python 3.8+
- MySQL Server
- CUDA (tùy chọn, cho GPU acceleration)

### Bước 1: Clone Repository
```bash
git clone <repository-url>
cd eye-recognition-service
```

### Bước 2: Tạo Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# hoặc
venv\Scripts\activate     # Windows
```

### Bước 3: Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Cấu Hình Environment
Tạo file `.env` và cấu hình các biến môi trường:
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_NAME=eye_recognition
DB_USER=your_username
DB_PASSWORD=your_password

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Model Configuration
MODEL_PATH=./models
UPLOAD_PATH=./uploads
```

### Bước 5: Khởi Chạy Database
Đảm bảo MySQL server đang chạy và tạo database:
```sql
CREATE DATABASE eye_recognition;
```

### Bước 6: Chạy Ứng Dụng
```bash
python main.py
```

Server sẽ khởi chạy tại `http://localhost:8000`

## API Endpoints

### Thông Tin Dịch Vụ
- `GET /` - Thông tin cơ bản về dịch vụ
- `GET /health` - Kiểm tra trạng thái sức khỏe của dịch vụ

### Quản Lý Mô Hình
- `POST /models` - Tải lên mô hình mới
- `GET /models` - Lấy danh sách mô hình
- `GET /models/{id}` - Lấy thông tin chi tiết mô hình
- `DELETE /models/{id}` - Xóa mô hình

### Quản Lý Dữ Liệu
- `POST /data` - Tải lên dữ liệu huấn luyện
- `GET /data` - Lấy danh sách dữ liệu
- `GET /data/{id}` - Xem chi tiết dữ liệu

### Huấn Luyện
- `POST /train` - Bắt đầu huấn luyện mô hình
- `GET /train/history` - Lịch sử huấn luyện
- `GET /train/{id}` - Chi tiết phiên huấn luyện

## Cấu Trúc Thư Mục

```
eye-recognition-service/
├── controller/              # API Controllers
│   ├── EyeDetectionModelController.py
│   ├── DetectEyeDataController.py
│   ├── DetectEyeDataTrainController.py
│   └── TrainDetectionHistoryController.py
├── dao/                     # Data Access Objects
├── entity/                  # Data Models
│   ├── EyeDetectionModel.py
│   ├── DetectEyeData.py
│   ├── DetectEyeDataTrain.py
│   └── TrainDetectionHistory.py
├── datasets/                # Training datasets
├── models/                  # Trained models
├── runs/                    # Training outputs
├── db_connection.py         # Database connection
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables
```

## Sử Dụng

### 1. Tải Lên Dữ Liệu Huấn Luyện
```bash
curl -X POST "http://localhost:8000/data" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@training_data.zip"
```

### 2. Bắt Đầu Huấn Luyện
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 100, "batch_size": 16}'
```

### 3. Kiểm Tra Trạng Thái Huấn Luyện
```bash
curl -X GET "http://localhost:8000/train/history"
```

## Đóng Góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add some amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## Hỗ Trợ

Nếu bạn gặp vấn đề hoặc có câu hỏi, vui lòng:
1. Kiểm tra [Issues](../../issues) để xem vấn đề đã được báo cáo chưa
2. Tạo issue mới nếu cần thiết
3. Liên hệ với team phát triển

## Changelog

### v1.0.0
- Phiên bản đầu tiên
- Hỗ trợ huấn luyện mô hình YOLO
- API quản lý mô hình và dữ liệu
- Giao diện web cơ bản

---

**Lưu ý**: Đây là dự án đang trong giai đoạn phát triển. Vui lòng kiểm tra thường xuyên để cập nhật các tính năng mới. 