# QUY TRÌNH XỬ LÝ VÂN TAY CHI TIẾT - TỪNG BƯỚC

> **Mục đích**: Giải thích chi tiết từng bước xử lý ảnh vân tay từ ảnh gốc đến kết quả cuối cùng

---

## 📊 TỔNG QUAN PIPELINE

Khi bạn nhập một ảnh vân tay vào hệ thống, ảnh sẽ đi qua **8 bước xử lý** để từ ảnh gốc → ảnh kết quả:

```
Ảnh Gốc → Chuẩn hóa → Phân đoạn → Định hướng → Lọc Gabor → Làm mỏng → Minutiae → Singularities
```

Dưới đây là **ĐẦY ĐỦ 8 BƯỚC** với ảnh minh họa thực tế:

![Pipeline đầy đủ](./output/DB3/0.png)

---

## BƯỚC 1: ẢNH GỐC (Original Image)

**📍 Vị trí**: Hàng 1, Cột 1

**Mã code**:
```python
input_img = cv.imread(path, 0)  # Đọc ảnh dạng grayscale
```

### 🎯 Mục đích
- Đọc ảnh vân tay gốc từ file
- Chuyển sang ảnh xám (grayscale) để dễ xử lý

### 📋 Đặc điểm
- **Độ sâu**: 8-bit grayscale (0-255)
- **Kích thước**: Thường 256x256 hoặc 320x240 pixels
- **Vấn đề**: 
  - Có nhiễu từ cảm biến
  - Độ sáng không đồng đều (do lực ấn ngón tay khác nhau)
  - Có vùng nền (background) không phải vân tay

---

## BƯỚC 2: CHUẨN HÓA (Normalization)

**📍 Vị trí**: Hàng 1, Cột 2

**Mã code**:
```python
normalized_img = normalize(input_img.copy(), float(100), float(100))
```

### 🎯 Mục đích
- **Loại bỏ nhiễu** từ cảm biến
- **Chuẩn hóa độ tương phản** - làm cho ảnh có độ sáng đồng đều
- **Loại bỏ ảnh hưởng** của lực ấn ngón tay khác nhau

### 🔬 Công thức toán học
```
I_norm(x,y) = M₀ + sqrt[ V₀ × (I(x,y) - M)² / V ]
```

Trong đó:
- `M₀ = 100`: Mean (trung bình) mong muốn
- `V₀ = 100`: Variance (phương sai) mong muốn
- `M`: Mean của ảnh gốc
- `V`: Variance của ảnh gốc
- `I(x,y)`: Giá trị pixel tại vị trí (x,y)

### 📊 Kết quả
- Ảnh có **độ tương phản đồng đều**
- **Giảm nhiễu** từ cảm biến
- Ảnh vẫn còn vùng nền (chưa tách vùng vân tay)

### 💡 Tại sao cần bước này?
Khi quét vân tay:
- Ngón tay ấn nhẹ → ảnh mờ
- Ngón tay ấn mạnh → ảnh tối
→ Chuẩn hóa giúp **đồng nhất** tất cả ảnh về cùng một chuẩn

---

## BƯỚC 3: PHÂN ĐOẠN (Segmentation)

**📍 Vị trí**: Hàng 1, Cột 3

**Mã code**:
```python
(segmented_img, normim, mask) = create_segmented_and_variance_images(
    normalized_img, block_size=16, threshold=0.2
)
```

### 🎯 Mục đích
- **Tách vùng vân tay** (Region of Interest - ROI) khỏi vùng nền
- **Loại bỏ vùng nền** màu đen (background)
- Tạo **mask** để đánh dấu vùng có vân tay

### 🔬 Cách hoạt động
1. **Chia ảnh thành các block** 16×16 pixels
2. Với mỗi block, tính **variance (phương sai)**:
   ```
   var = sqrt( sum[(pixel - mean)²] / n )
   ```
3. Nếu `var > threshold`:
   - Block có vân tay (giữ lại)
4. Nếu `var < threshold`:
   - Block là nền (loại bỏ → màu đen)

### 📊 Kết quả
- Vùng **có vân tay**: giữ nguyên (màu xám)
- Vùng **nền**: chuyển sang **màu đen**
- Tạo ra **mask** (mặt nạ) nhị phân:
  - 1 = vùng vân tay
  - 0 = vùng nền

### 💡 Tại sao cần bước này?
- Vùng nền **không có thông tin** → xử lý sẽ **lãng phí thời gian**
- Tập trung xử lý **chỉ vùng có vân tay** → nhanh hơn, chính xác hơn

---

## BƯỚC 4: TÍNH HƯỚNG VÂN (Orientation Estimation)

**📍 Vị trí**: Hàng 1, Cột 4

**Mã code**:
```python
angles = orientation.calculate_angles(normalized_img, W=16, smoth=False)
orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=16)
```

### 🎯 Mục đích
- Tính **góc hướng** của các đường vân tay
- Xác định vân tay **chạy theo hướng nào** tại mỗi vị trí

### 🔬 Cách hoạt động
1. **Chia ảnh thành block** 16×16
2. Với mỗi block, dùng **Sobel Operator** tính gradient:
   ```
   Gx = ∂I/∂x  (đạo hàm theo trục x)
   Gy = ∂I/∂y  (đạo hàm theo trục y)
   ```
3. Tính góc hướng:
   ```
   θ = 0.5 × arctan(2×Gxy / (Gxx - Gyy))
   ```

### 📊 Kết quả (Hình ảnh màu xanh-đỏ)
- **Các đường màu xanh/đỏ**: thể hiện **hướng** của vân tay tại mỗi block
- Mỗi đường = **vector hướng** của các đường vân
- Ảnh trông như **bản đồ dòng chảy** (flow map)

### 💡 Tại sao cần bước này?
- Biết **hướng vân** giúp lọc Gabor hoạt động **chính xác hơn**
- Các đường vân **không song song** → cần biết hướng để xử lý đúng

---

## BƯỚC 5: LỌC GABOR (Gabor Filtering)

**📍 Vị trí**: Hàng 2, Cột 1

**Mã code**:
```python
freq = ridge_freq(normim, mask, angles, block_size=16, 
                  kernel_size=5, minWaveLength=5, maxWaveLength=15)
gabor_img = gabor_filter(normim, angles, freq)
```

### 🎯 Mục đích
- **Làm rõ đường vân** (ridge enhancement)
- **Giảm nhiễu** trong ảnh
- **Làm đậm** các đường vân, làm **nhạt** vùng giữa các vân

### 🔬 Cách hoạt động

#### Bước 5.1: Tính tần số vân (Ridge Frequency)
```python
freq = ridge_freq(...)
```
- Xác định **mật độ đường vân** (bao nhiêu đường vân/pixel)
- Tính trong miền **Wavelet Domain**
- Kết quả: mỗi block có một giá trị `freq` (tần số)

#### Bước 5.2: Áp dụng bộ lọc Gabor
```python
gabor_img = gabor_filter(normim, angles, freq)
```

**Công thức Gabor Filter**:
```
G(x,y,θ,f) = exp(-πf² [(x'²/σx²) + (y'²/σy²)]) × cos(2πfx')

Trong đó:
x' = x×cos(θ) + y×sin(θ)
y' = -x×sin(θ) + y×cos(θ)
```

- `θ`: góc hướng (từ bước 4)
- `f`: tần số (từ bước 5.1)
- `σx, σy`: độ rộng bộ lọc

### 📊 Kết quả
- Ảnh **đen-trắng rõ nét**
- **Đường vân màu trắng** (sáng)
- **Vùng giữa các vân màu đen** (tối)
- **Nhiễu giảm** đáng kể

### 💡 Tại sao dùng Gabor?
- Gabor là bộ lọc **theo hướng** → chỉ lọc theo hướng vân tay
- Gabor có **tần số** → khớp với tần số vân tay
- Kết quả: **chỉ giữ lại đường vân**, loại bỏ nhiễu

---

## BƯỚC 6: LÀM MỎNG (Thinning/Skeletonization)

**📍 Vị trí**: Hàng 2, Cột 2

**Mã code**:
```python
thin_image = skeletonize(gabor_img)
```

### 🎯 Mục đích
- **Thu gọn đường vân** về độ dày **1 pixel**
- Tạo ra **"bộ xương"** (skeleton) của vân tay
- Dễ dàng phát hiện **điểm minutiae**

### 🔬 Cách hoạt động
Sử dụng **thuật toán Zhang-Suen**:

1. **Lặp lại** cho đến khi không còn pixel nào bị xóa:
2. **Quét toàn bộ ảnh**, với mỗi pixel kiểm tra:
   - Số lượng pixel láng giềng màu trắng (2 ≤ n ≤ 6)
   - Số lần chuyển từ đen → trắng (= 1)
   - Các điều kiện khác để đảm bảo **không làm đứt đường**
3. **Xóa pixel** nếu thỏa mãn tất cả điều kiện
4. Lặp lại cho đến khi **ổn định**

### 📊 Kết quả
- Đường vân **chỉ dày 1 pixel**
- **Giữ nguyên cấu trúc** của vân tay
- Dễ dàng **phát hiện điểm đặc biệt**

### 💡 Tại sao cần bước này?
- Đường vân dày → **khó phát hiện** điểm kết thúc/phân nhánh
- Đường vân mỏng (1 pixel) → dễ dàng **đếm pixel láng giềng**

---

## BƯỚC 7: PHÁT HIỆN MINUTIAE

**📍 Vị trí**: Hàng 2, Cột 3

**Mã code**:
```python
minutias = calculate_minutiaes(thin_image)
```

### 🎯 Mục đích
- Phát hiện **điểm đặc trưng** (minutiae) của vân tay
- Đây là **đặc điểm độc nhất** để nhận dạng con người

### 🔬 Cách hoạt động - Phương pháp Crossing Number (CN)

#### 1. Với mỗi pixel trắng, đếm số pixel láng giềng
```
P9  P2  P3
P8  P1  P4
P7  P6  P5
```

#### 2. Tính Crossing Number (CN)
```
CN = 0.5 × sum|Pi - Pi+1|  (i = 1→8, P9 = P1)
```

#### 3. Phân loại dựa trên CN
- **CN = 1**: **Điểm kết thúc** (Termination)
  - Đường vân **đứt tại đây**
  - Màu đỏ trong ảnh
  
- **CN = 3**: **Điểm phân nhánh** (Bifurcation)
  - Đường vân **tách thành 2-3 nhánh**
  - Màu xanh lá trong ảnh

### 📊 Kết quả
- **Chấm đỏ**: điểm kết thúc (termination)
- **Chấm xanh lá**: điểm phân nhánh (bifurcation)
- Mỗi điểm lưu: `[loại, x, y, góc_hướng]`

### 💡 Tại sao minutiae quan trọng?
- Mỗi người có **khoảng 40-100 minutiae** trên 1 ngón tay
- **Vị trí và hướng** của minutiae là **DUY NHẤT** cho mỗi người
- → Dùng để **nhận dạng** và **xác thực**

---

## BƯỚC 8: PHÁT HIỆN SINGULARITIES

**📍 Vị trí**: Hàng 2, Cột 4

**Mã code**:
```python
singularities_img = calculate_singularities(
    thin_image, angles, tolerance=1, block_size=16, mask=mask
)
```

### 🎯 Mục đích
- Phát hiện **điểm kỳ dị** (Singularities)
- 2 loại điểm đặc biệt:
  - **Core**: điểm trung tâm (vân tròn)
  - **Delta**: điểm tam giác (vân tách ba)

### 🔬 Cách hoạt động - Phương pháp Poincaré Index

#### 1. Tạo vòng tròn quanh mỗi điểm
```
Chọn tâm tại (x,y)
Tạo vòng tròn bán kính R
Lấy N điểm trên vòng tròn
```

#### 2. Tính tổng thay đổi góc
```
Poincaré Index = (1/2π) × sum[Δθi]
```

Trong đó `Δθi` là sự **thay đổi góc** giữa 2 điểm liên tiếp

#### 3. Phân loại
- **Index ≈ +0.5**: **Điểm Core** (màu cam)
  - Vân tay **xoáy tròn** về điểm này
  
- **Index ≈ -0.5**: **Điểm Delta** (không hiển thị rõ)
  - Vân tay **phân ba hướng**

### 📊 Kết quả
- **Chấm cam**: điểm Core (trung tâm xoáy vân)
- Thường có **1-2 điểm Core** trên mỗi vân tay

### 💡 Tại sao singularities quan trọng?
- Giúp **phân loại vân tay**:
  - Có 2 delta → Whorl (xoáy)
  - Có 1 delta → Loop (vòng)
  - Không có delta → Arch (cung)
- Dùng làm **điểm tham chiếu** để căn chỉnh ảnh

---

## 🔄 SAU KHI XỬ LÝ: SO KHỚP VÀ TÌM KIẾM

Sau khi có **danh sách minutiae**, hệ thống thực hiện:

### 1. Lưu đặc trưng
```python
minutiae_list = [
    [type, x1, y1, angle1],
    [type, x2, y2, angle2],
    ...
]
```

### 2. So khớp với database
```python
for each_image in database:
    match_count = 0
    for mi in input_minutiae:
        for mt in database_minutiae:
            # Tính khoảng cách không gian
            sd = sqrt((mi.x - mt.x)² + (mi.y - mt.y)²)
            
            # Tính khoảng cách góc
            dd = min(|mi.angle - mt.angle|, 2π - |mi.angle - mt.angle|)
            
            # Kiểm tra điều kiện khớp
            if sd < 50 and dd < π/24 and mi.type == mt.type:
                match_count += 1
    
    # Lưu ảnh có match_count cao nhất
```

### 3. Trả về kết quả
- **Ảnh khớp nhất**: ảnh có **số minutiae khớp nhiều nhất**
- **Thời gian xử lý**: ~2-7 giây
- **Hiển thị ảnh**: ảnh vân tay được tìm thấy trong database

---

## 📈 TÓM TẮT TOÀN BỘ QUY TRÌNH

| Bước | Tên | Input | Output | Mục đích |
|------|-----|-------|--------|----------|
| 1 | Original | File ảnh | Ảnh grayscale | Đọc ảnh gốc |
| 2 | Normalization | Ảnh gốc | Ảnh chuẩn hóa | Loại nhiễu, đồng nhất độ sáng |
| 3 | Segmentation | Ảnh chuẩn | Ảnh + mask vùng vân | Tách vùng vân tay |
| 4 | Orientation | Ảnh chuẩn | Ảnh + góc hướng | Tính hướng vân |
| 5 | Gabor Filter | Ảnh + hướng + tần số | Ảnh làm rõ vân | Làm rõ đường vân |
| 6 | Thinning | Ảnh Gabor | Ảnh vân mỏng 1px | Thu gọn vân |
| 7 | Minutiae | Ảnh mỏng | Danh sách minutiae | Trích đặc trưng |
| 8 | Singularities | Ảnh mỏng + hướng | Điểm Core/Delta | Tìm điểm đặc biệt |
| 9 | Matching | Minutiae input | Ảnh khớp nhất | So khớp database |

---

## 🎯 KẾT LUẬN

Hệ thống đã hoàn thành **ĐẦY ĐỦ 3 TIÊU CHÍ**:

### ✅ 1. Nâng cao chất lượng ảnh
- Bước 2: Normalization
- Bước 3: Segmentation
- Bước 4: Orientation
- Bước 5: Gabor Filtering
- Bước 6: Thinning

### ✅ 2. Trích chọn đặc trưng
- Bước 7: Minutiae Detection (Crossing Number)
- Bước 8: Singularities Detection (Poincaré Index)

### ✅ 3. Xác thực vân tay
- Bước 9: Matching với database
- Tìm ảnh vân tay khớp nhất

---

## 📚 THAM KHẢO

- [Fingerprint Enhancement Techniques](https://www.cse.iitk.ac.in/users/biometrics/)
- Zhang-Suen Thinning Algorithm
- Crossing Number Method (CN)
- Poincaré Index Method
- Gabor Filter Theory
