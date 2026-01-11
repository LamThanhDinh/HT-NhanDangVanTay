# Hệ thống Nhận dạng Vân tay (Fingerprint Recognition System)

Dự án xây dựng hệ thống nhận dạng và xác thực vân tay sử dụng các kỹ thuật xử lý ảnh và trích xuất đặc trưng minutiae.

## Mục tiêu

Hệ thống thực hiện 3 chức năng chính:
1. **Nâng cao chất lượng ảnh** - Tiền xử lý và làm sạch ảnh vân tay
2. **Trích chọn đặc trưng** - Phát hiện các điểm minutiae (đặc điểm độc nhất)
3. **Xác thực vân tay** - So khớp và tìm kiếm vân tay trong cơ sở dữ liệu

## Tính năng

### 1. Xử lý và Nâng cao Chất lượng Ảnh
- **Chuẩn hóa** (Normalization): Loại bỏ nhiễu và chuẩn hóa độ tương phản
- **Phân đoạn** (Segmentation): Tách vùng vân tay khỏi nền
- **Tính góc hướng** (Orientation): Xác định hướng của các đường vân
- **Tính tần số vân** (Frequency): Xác định mật độ đường vân
- **Lọc Gabor** (Gabor Filter): Làm rõ đường vân, giảm nhiễu
- **Làm mỏng** (Thinning/Skeletonization): Thu gọn đường vân về độ dày 1 pixel

### 2. Trích xuất Đặc trưng
- **Điểm Minutiae**: Phát hiện 2 loại minutiae
  - Điểm kết thúc (Termination)
  - Điểm phân nhánh (Bifurcation)
- **Điểm Kỳ dị** (Singularities): Phát hiện điểm core và delta
- **Vector đặc trưng**: Lưu vị trí (x, y) và góc hướng của mỗi minutiae

### 3. So khớp và Xác thực
- Tính khoảng cách Euclidean giữa các điểm minutiae
- Tính độ lệch góc hướng
- Tìm ảnh vân tay khớp nhất trong database

## Cấu trúc Dự án

```
btl/
├── data/                          # Module xử lý dữ liệu
│   ├── data_procesing.py         # Tiền xử lý dữ liệu
│   ├── normal_image.py           # Chuẩn hóa ảnh
│   ├── segmentation.py           # Phân đoạn ảnh
│   ├── orientation.py            # Tính góc hướng
│   ├── frequency.py              # Tính tần số vân
│   ├── gaborfilter.py            # Lọc Gabor
│   ├── torch_Dataset.py          # Dataset cho PyTorch
│   └── dataset/                  # Dữ liệu vân tay
│       ├── train/                # Dữ liệu huấn luyện
│       │   ├── DB1/
│       │   ├── DB2/
│       │   ├── DB3/
│       │   └── DB4/
│       ├── test/                 # Dữ liệu test
│       └── db_data.json          # Database đã xử lý
│
├── model/                         # Module mô hình
│   ├── minunate_detection.py    # Phát hiện minutiae
│   ├── calculate_distance.py    # Tính khoảng cách so khớp
│   ├── thinning.py              # Thuật toán làm mỏng
│   └── models_pretrain.py       # Mô hình đã huấn luyện
│
├── utils/                         # Các hàm tiện ích
│   ├── normalization.py          # Chuẩn hóa
│   ├── segmentation.py           # Phân đoạn
│   ├── orientation.py            # Góc hướng
│   ├── frequency.py              # Tần số
│   ├── gabor_filter.py           # Lọc Gabor
│   ├── skeletonize.py            # Làm mỏng
│   ├── crossing_number.py        # Phát hiện minutiae
│   └── poincare.py               # Phát hiện singularities
│
├── output/                        # Kết quả xử lý
│   └── DB3/                      # Ảnh đã xử lý từ DB3
│
├── finegerprint_pipline.py       # Pipeline xử lý batch
└── pipline.py                    # Pipeline tìm kiếm đơn lẻ
```

## Cài đặt

### 1. Yêu cầu
- Python 3.12+
- pip

### 2. Clone Repository

```bash
git clone <repository-url>
cd btl
```

### 3. Cài đặt Dependencies

```bash
pip install opencv-python numpy tqdm scikit-image torch torchvision pandas matplotlib
```

Hoặc nếu đã có virtual environment:

```bash
# Windows
.venv\Scripts\activate
pip install opencv-python numpy tqdm scikit-image torch torchvision pandas matplotlib
```

## CÁCH CHẠY ĐƠN GIẢN NHẤT

### Bước 1: Cài đặt thư viện
```bash
pip install opencv-python numpy tqdm scikit-image torch torchvision pandas matplotlib
```

### Bước 2: Chạy chương trình
```bash
python pipline.py
```

### Bước 3: Nhập đường dẫn ảnh
Khi chương trình yêu cầu, nhập đường dẫn ảnh vân tay:
```
Mời nhập link ảnh: ./data/dataset/test/DB1/101_1.tif
```

### Kết quả
- Hiển thị thời gian xử lý (khoảng 6-7 giây)
- Đường dẫn ảnh vân tay khớp nhất trong database
- Cửa sổ hiển thị ảnh vân tay tìm được

### Ví dụ kết quả
```
6.665056467056274
./data/dataset/train/DB2/109_7.tif
```

---

## Cách sử dụng nâng cao

### Xử lý Batch Toàn bộ Dataset

Xử lý và lưu kết quả cho tất cả ảnh trong thư mục:

```bash
python finegerprint_pipline.py
```

**Input:** Ảnh trong `./data/dataset/train/DB3/`

**Output:** Kết quả được lưu vào `./output/DB3/`

Mỗi ảnh kết quả bao gồm 8 bước xử lý:
1. Ảnh gốc
2. Ảnh chuẩn hóa
3. Ảnh phân đoạn
4. Ảnh góc hướng
5. Ảnh sau Gabor filter
6. Ảnh làm mỏng
7. Ảnh minutiae
8. Ảnh singularities

## 🔬 Thuật toán và Kỹ thuật

### Pipeline Xử lý

```
Input Image
    ↓
Normalization (Chuẩn hóa)
    ↓
Segmentation (Phân đoạn ROI)
    ↓
Orientation Estimation (Tính góc hướng)
    ↓
Frequency Estimation (Tính tần số vân)
    ↓
Gabor Filtering (Lọc Gabor)
    ↓
Thinning/Skeletonization (Làm mỏng)
    ↓
Minutiae Extraction (Trích xuất minutiae)
    ↓
Singularity Detection (Phát hiện singularities)
    ↓
Matching (So khớp)
```

### Các Kỹ thuật Sử dụng

1. **Sobel Operator**: Tính gradient để xác định góc hướng vân
2. **Ridge Frequency**: Xác định tần số đường vân trong miền Wavelet
3. **Gabor Filter**: Lọc theo hướng và tần số của vân
4. **Zhang-Suen Algorithm**: Làm mỏng ảnh binary
5. **Crossing Number (CN)**: Phát hiện minutiae
   - CN = 1: Điểm kết thúc (Termination)
   - CN = 3: Điểm phân nhánh (Bifurcation)
6. **Poincaré Index**: Phát hiện điểm core và delta

### So khớp Vân tay

Sử dụng công thức khoảng cách:

- **Khoảng cách không gian (sd)**: 
  ```
  sd = √[(x₁-x₂)² + (y₁-y₂)²]
  ```

- **Khoảng cách góc (dd)**:
  ```
  dd = min(|θ₁-θ₂|, 2π - |θ₁-θ₂|)
  ```

Điều kiện khớp:
- `sd < 50` pixels
- `dd < π/24` radians (~7.5°)
- Cùng loại minutiae

## Dataset

Dự án sử dụng dataset vân tay chuẩn:
- **DB1, DB2, DB3, DB4**: Các database khác nhau
- Mỗi database chứa nhiều mẫu vân tay
- Format: `.tif` (TIFF images)

## Xử lý Lỗi

### Lỗi thường gặp:

**1. `ModuleNotFoundError: No module named 'cv2'`**
```bash
pip install opencv-python
```

**2. `AttributeError: module 'numpy' has no attribute 'int'`**
- Đã được sửa trong code (thay `np.int` → `int`)

**3. `IndexError: list index out of range`**
- Đã thêm kiểm tra an toàn cho list points

**4. `TypeError: unsupported operand type(s) for -: 'float' and 'NoneType'`**
- Đã thêm kiểm tra None cho orientation values

## Kết quả

Hệ thống có thể:
- Xử lý và nâng cao chất lượng ảnh vân tay
- Trích xuất chính xác các điểm minutiae
- So khớp và tìm kiếm vân tay trong database
- Thời gian xử lý: ~6-7 giây/ảnh
- Hiển thị trực quan các bước xử lý

## Ứng dụng

Dự án này đáp ứng các yêu cầu:
1. Nâng cao chất lượng ảnh trong hệ thống nhận dạng vân tay
2. Trích chọn đặc trưng trong hệ thống nhận dạng vân tay
3. Xác thực dựa trên vân tay

## Tham khảo

- [Fingerprint Enhancement and Minutiae Extraction](https://www.cse.iitk.ac.in/users/biometrics/)
- [Crossing Number Method for Minutiae Detection](https://ieeexplore.ieee.org/)
- [Gabor Filters for Fingerprint Enhancement](https://www.sciencedirect.com/)

## Ghi chú

- Thông báo "loi tim diem thu 2" là debug message, không ảnh hưởng kết quả
- Một số minutiae có thể có góc hướng `None` khi không đủ điểm để tính vector
- Hệ thống tự động bỏ qua các trường hợp đặc biệt

## Đóng góp

Dự án được xây dựng cho môn học Hệ Cơ sở Dữ liệu Đa phương tiện và Phân tán.

---

**Lưu ý**: Đảm bảo đã cài đặt đầy đủ dependencies trước khi chạy chương trình.
