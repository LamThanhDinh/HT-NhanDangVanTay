# 🌐 HƯỚNG DẪN CHẠY GIAO DIỆN WEB ĐƠN GIẢN

## ✅ SIÊU ĐƠN GIẢN - CHỈ 2 BƯỚC!

### Bước 1️⃣: Chạy Server
```bash
python server.py
```

### Bước 2️⃣: Mở Trình duyệt
Truy cập: **http://localhost:8000**

---

## 🎯 CÁCH SỬ DỤNG

1. **Xem Demo**: Click nút "🎨 Xem Demo" để xem ngay với ảnh mẫu
2. **Upload ảnh**: Kéo thả hoặc chọn file ảnh vân tay của bạn, sau đó click "📤 Upload và Xử lý"
3. **Xem kết quả**: Cuộn xuống xem **9 bước xử lý** chi tiết

---

## 📋 KHÔNG CẦN CÀI FLASK!

Server này dùng `http.server` có sẵn trong Python, **không cần cài thêm gì cả**!

Chỉ cần các thư viện xử lý ảnh:
```bash
pip install opencv-python numpy scikit-image
```

---

## 🎨 9 BƯỚC XỬ LÝ

1. ✅ Ảnh gốc
2. ✅ Chuẩn hóa (Normalization)
3. ✅ Phân đoạn (Segmentation)
4. ✅ Định hướng (Orientation)
5. ✅ Phân tích tần số (Frequency)
6. ✅ Lọc Gabor (Gabor Filter)
7. ✅ Làm mỏng (Skeletonization)
8. ✅ Điểm đặc trưng (Minutiae)
9. ✅ Điểm kỳ dị (Singularities)

Mỗi bước đều có:
- 🖼️ Ảnh kết quả
- 📝 Giải thích chi tiết
- ⚙️ Thông số kỹ thuật

---

## 🚀 THẬT SỰ RẤT ĐƠN GIẢN!

**Không phải Flask phức tạp**  
**Chỉ là HTML + Python server đơn giản**  
**Mở trình duyệt là chạy ngay!**

---

## 🛑 DỪNG SERVER

Nhấn `Ctrl + C` trong terminal để dừng server

---

## 💡 GHI CHÚ

- Server chạy ở port 8000 (có thể đổi trong file server.py)
- File HTML nằm ở thư mục gốc: `index.html`
- Ảnh demo tự động load từ thư mục `data/dataset/`

**Enjoy! 🎉**
