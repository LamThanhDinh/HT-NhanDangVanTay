import http.server
import socketserver
import json
import base64
import cv2 as cv
import numpy as np
from urllib.parse import parse_qs, urlparse
import os
from io import BytesIO

from utils.normalization import normalize
from utils.segmentation import create_segmented_and_variance_images
from utils import orientation
from utils.frequency import ridge_freq
from utils.gabor_filter import gabor_filter
from utils.skeletonize import skeletonize
from utils.crossing_number import calculate_minutiaes
from utils.poincare import calculate_singularities

PORT = 8080

def image_to_base64(img):
    """Chuyển ảnh numpy array sang base64 string"""
    # Đảm bảo ảnh là uint8
    if img.dtype != np.uint8:
        # Normalize về range 0-255 nếu cần
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Convert sang RGB nếu là grayscale
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    
    _, buffer = cv.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def process_fingerprint(input_img):
    """Xử lý ảnh vân tay và trả về từng bước (8 bước + composite)"""
    block_size = 16
    steps = []
    
    # BƯỚC 1: Ảnh gốc
    steps.append({
        'step': 1,
        'name': '🖼️ Ảnh Gốc (Original Image)',
        'description': 'Đây là ảnh vân tay ban đầu được đọc từ cảm biến hoặc file. Ảnh được chuyển sang dạng grayscale (ảnh xám) để chuẩn bị cho các bước xử lý tiếp theo.',
        'image': image_to_base64(input_img),
        'details': f'📏 Kích thước: {input_img.shape[1]} × {input_img.shape[0]} pixels | 🎨 Format: 8-bit Grayscale'
    })
    
    # BƯỚC 2: Chuẩn hóa
    normalized_img = normalize(input_img.copy(), float(100), float(100))
    steps.append({
        'step': 2,
        'name': '⚖️ Chuẩn hóa (Normalization)',
        'description': 'Điều chỉnh độ sáng và độ tương phản của ảnh về một chuẩn thống nhất (mean=100, variance=100). Loại bỏ ảnh hưởng của lực ấn ngón tay, độ ẩm da, và điều kiện chiếu sáng khác nhau giữa các lần quét.',
        'image': image_to_base64(normalized_img),
        'details': '📊 Mean = 100 | 📈 Variance = 100 | 🎯 Mục đích: Chuẩn hóa cường độ pixel'
    })
    
    # BƯỚC 3: Phân đoạn
    segmented_img, normim, mask = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
    steps.append({
        'step': 3,
        'name': '✂️ Phân đoạn (Segmentation)',
        'description': 'Tách vùng có vân tay (foreground/ROI) ra khỏi vùng nền trống (background). Sử dụng phương pháp variance-based: vùng có vân tay có độ biến thiên cao (ridge-valley alternation), vùng nền có độ biến thiên thấp.',
        'image': image_to_base64(segmented_img),
        'details': f'🔲 Block size: {block_size}×{block_size} pixels | 📉 Threshold: 0.2 × std(image) | ✨ Morphological refinement'
    })
    
    # BƯỚC 4: Định hướng
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
    steps.append({
        'step': 4,
        'name': '🧭 Định hướng (Orientation Field)',
        'description': 'Tính toán hướng (góc) của các đường vân tại mỗi vùng block. Các đường màu xanh lá cây chỉ hướng vân tay chạy. Thông tin này rất quan trọng cho bước lọc Gabor.',
        'image': image_to_base64(orientation_img),
        'details': '🎨 Màu xanh = hướng vân | 📐 Phương pháp: Gradient-based (Sobel) | 🔢 Block size: 16×16',
        'legend': '<span style="color: #00FF00; font-weight: bold;">━━━ Hướng vân (Green)</span>'
    })
    
    # Tính frequency (không hiển thị riêng, chỉ dùng cho Gabor)
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    
    # BƯỚC 5: Lọc Gabor
    gabor_img = gabor_filter(normim, angles, freq)
    steps.append({
        'step': 5,
        'name': '🎛️ Lọc Gabor (Gabor Filter)',
        'description': 'Bộ lọc quan trọng nhất! Gabor filter kết hợp thông tin về hướng vân (orientation) và tần số vân (frequency) để làm nổi bật đường vân, giảm nhiễu, và kết nối các đường vân bị đứt gãy. Ảnh sau bước này có ridge rõ nét nhất.',
        'image': image_to_base64(gabor_img),
        'details': '🎚️ Filter kết hợp orientation + frequency | ⚙️ Parameters: σₓ=σᵧ=0.65/f | ⭐ Bước quan trọng nhất!'
    })
    
    # BƯỚC 6: Làm mỏng
    thin_image = skeletonize(gabor_img)
    steps.append({
        'step': 6,
        'name': '🦴 Làm mỏng (Skeletonization)',
        'description': 'Thu gọn các đường vân từ độ dày 5-10 pixels xuống chỉ còn 1 pixel (skeleton). Điều này giúp xác định chính xác vị trí các điểm đặc trưng minutiae ở bước tiếp theo. Sử dụng thuật toán Zhang-Suen.',
        'image': image_to_base64(thin_image),
        'details': '⚙️ Algorithm: Zhang-Suen | 📏 Ridge width: 1 pixel | ✅ Topology preserved'
    })
    
    # BƯỚC 7: Điểm đặc trưng (Minutiae)
    minutias_img = calculate_minutiaes(thin_image)
    steps.append({
        'step': 7,
        'name': '🔴🟢 Minutiae (Điểm đặc trưng)',
        'description': 'Phát hiện các điểm đặc trưng minutiae - những điểm quan trọng nhất để nhận dạng vân tay. Có 2 loại: Ridge Ending (điểm kết thúc - màu ĐỎ) và Bifurcation (điểm phân nhánh - màu XANH LÁ). Sử dụng phương pháp Crossing Number.',
        'image': image_to_base64(minutias_img),
        'details': '🔴 Ridge Ending (CN=1) | 🟢 Bifurcation (CN=3) | 📊 Trung bình: 40-60 minutiae/ảnh | 💾 Lưu: (type, x, y, θ)',
        'legend': '<span style="color: red; font-weight: bold;">● Ridge Ending (Đỏ)</span> &nbsp;&nbsp; <span style="color: green; font-weight: bold;">● Bifurcation (Xanh)</span>'
    })
    
    # BƯỚC 8: Điểm kỳ dị (Singularities)
    singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)
    steps.append({
        'step': 8,
        'name': '🟠 Singularities (Điểm kỳ dị)',
        'description': 'Phát hiện các điểm kỳ dị (Core, Delta, Whorl) - những điểm mà hướng vân tay thay đổi đột ngột. Core là tâm xoáy (ô vuông CAM), Delta là điểm tam giác (ô vuông ĐỎ), Whorl là điểm xoáy (ô vuông TÍM). Bước này CHỈ ĐỂ TRỰC QUAN HÓA, không dùng cho matching.',
        'image': image_to_base64(singularities_img),
        'details': '🟧 Core (Cam) | 🟥 Delta (Đỏ) | 🟪 Whorl (Tím) | ℹ️ Chỉ để hiển thị, không dùng matching',
        'legend': '<span style="display:inline-block;width:18px;height:18px;border:2px solid orange;vertical-align:middle;margin-right:4px;"></span> Core (Cam) &nbsp;&nbsp;'
                  '<span style="display:inline-block;width:18px;height:18px;border:2px solid red;vertical-align:middle;margin-right:4px;"></span> Delta (Đỏ) &nbsp;&nbsp;'
                  '<span style="display:inline-block;width:18px;height:18px;border:2px solid purple;vertical-align:middle;margin-right:4px;"></span> Whorl (Tím)'
    })
    
    # TẠO COMPOSITE IMAGE (ghép 8 ảnh thành 2 hàng × 4 cột)
    output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias_img, singularities_img]
    
    # Convert tất cả sang RGB nếu là grayscale
    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
    
    # Ghép 2 hàng: hàng 1 (ảnh 0-3), hàng 2 (ảnh 4-7)
    composite_img = np.concatenate([
        np.concatenate(output_imgs[:4], 1),  # Hàng 1: Original, Norm, Seg, Orient
        np.concatenate(output_imgs[4:], 1)   # Hàng 2: Gabor, Thin, Minutiae, Singularities
    ]).astype(np.uint8)
    
    # Thêm composite image vào cuối
    steps.append({
        'step': 9,  # Composite ở cuối cùng
        'name': '🎨 KẾT QUẢ TỔNG HỢP (Composite)',
        'description': 'Tổng hợp 8 bước xử lý trong 1 ảnh duy nhất. Hàng trên: Original, Normalization, Segmentation, Orientation. Hàng dưới: Gabor Filter, Skeletonization, Minutiae, Singularities.',
        'image': image_to_base64(composite_img),
        'details': '📐 Layout: 2 hàng × 4 cột | 🖼️ Tất cả 8 bước trong 1 ảnh | 💾 Dễ lưu và so sánh'
    })
    
    return steps

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/demo':
            self.handle_demo()
            return
        elif self.path == '/':
            self.path = '/index.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        if self.path == '/api/process':
            self.handle_upload()
            return
        self.send_response(404)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
    def handle_demo(self):
        """Xử lý ảnh demo"""
        try:
            # Tìm ảnh demo
            demo_paths = [
                './data/dataset/test/DB3',
                './data/dataset/test/DB1',
                './data/dataset/train/DB3',
                './data/dataset/train/DB1',
            ]
            
            demo_img_path = None
            for path in demo_paths:
                if os.path.exists(path):
                    files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.bmp', '.tif'))]
                    if files:
                        demo_img_path = os.path.join(path, files[0])
                        break
            
            if demo_img_path is None:
                self.send_json_response({'error': 'Không tìm thấy ảnh demo'}, 404)
                return
            
            input_img = cv.imread(demo_img_path, cv.IMREAD_GRAYSCALE)
            steps = process_fingerprint(input_img)
            
            self.send_json_response({'success': True, 'steps': steps})
        except Exception as e:
            self.send_json_response({'error': str(e)}, 500)
    
    def handle_upload(self):
        """Xử lý upload ảnh"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse multipart form data - tìm boundary
            content_type = self.headers.get('Content-Type', '')
            if 'boundary=' not in content_type:
                self.send_json_response({'error': 'Invalid content type'}, 400)
                return
                
            boundary = content_type.split('boundary=')[1].encode()
            parts = post_data.split(b'--' + boundary)
            
            image_data = None
            for part in parts:
                if b'Content-Type: image' in part or b'filename=' in part:
                    try:
                        # Tách header và data
                        headers_and_data = part.split(b'\r\n\r\n', 1)
                        if len(headers_and_data) == 2:
                            image_data = headers_and_data[1].rstrip(b'\r\n')
                            if len(image_data) > 100:  # Đảm bảo có data
                                break
                    except:
                        continue
            
            if image_data is None or len(image_data) < 100:
                self.send_json_response({'error': 'Không tìm thấy ảnh trong request'}, 400)
                return
            
            # Decode ảnh
            nparr = np.frombuffer(image_data, np.uint8)
            input_img = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE)
            
            if input_img is None:
                self.send_json_response({'error': 'Không thể đọc ảnh. Vui lòng upload file ảnh hợp lệ'}, 400)
                return
            
            steps = process_fingerprint(input_img)
            self.send_json_response({'success': True, 'steps': steps})
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json_response({'error': f'Lỗi server: {str(e)}'}, 500)
    
    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

if __name__ == '__main__':

    import sys
    def visualize_local(image_path):
        import matplotlib.pyplot as plt
        import re
        print("\n" + "=" * 70)
        print("🔬 PHÂN TÍCH PIPELINE VÂN TAY - LOCAL MODE")
        print("=" * 70)
        print(f"📸 Đang xử lý: {image_path}")
        input_img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if input_img is None:
            print(f"❌ Không thể đọc ảnh: {image_path}")
            return
        steps = process_fingerprint(input_img)
        composite_base64 = steps[-1]['image']
        img_data = re.sub('^data:image/.+;base64,', '', composite_base64)
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        composite_img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        composite_rgb = cv.cvtColor(composite_img, cv.COLOR_BGR2RGB)
        # Hiển thị legend phía trên ảnh
        import matplotlib.patches as mpatches
        plt.figure(figsize=(20, 10))
        plt.title('Pipeline Xử Lý Vân Tay - 8 Bước\nHàng 1: (1) Original | (2) Normalized | (3) Segmented | (4) Orientation\nHàng 2: (5) Gabor | (6) Thinning | (7) Minutiae | (8) Singularities', fontsize=14, pad=30)
        plt.imshow(composite_rgb)
        plt.axis('off')
        # Chú thích bằng patch vuông màu
        legend_handles = [
            mpatches.Patch(color='red', label='Minutiae: 🔴 Ridge Ending'),
            mpatches.Patch(color='green', label='🟢 Bifurcation'),
            mpatches.Patch(color=(1.0, 0.65, 0), label='Singularities: 🟧 Core (Cam)'),
            mpatches.Patch(color='magenta', label='🟪 Whorl (Tím)'),
            mpatches.Patch(color='darkred', label='🟥 Delta (Đỏ)'),
        ]
        plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=13, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
        # Hiển thị từng bước riêng lẻ
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        for i, step_data in enumerate(steps[:8]):
            img_data = re.sub('^data:image/.+;base64,', '', step_data['image'])
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_COLOR)
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(step_data['name'], fontsize=10)
            axes[i].axis('off')
        # Chú thích bằng patch vuông màu cho từng bước
        fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=13, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    if len(sys.argv) > 1 and sys.argv[1] != 'server':
        visualize_local(sys.argv[1])
    else:
        Handler = MyHTTPRequestHandler
        print("=" * 70)
        print("🚀 KHỞI ĐỘNG SERVER XỬ LÝ VÂN TAY")
        print("=" * 70)
        print(f"📍 Mở trình duyệt và truy cập: http://localhost:{PORT}")
        print("=" * 70)
        print("💡 Nhấn Ctrl+C để dừng server")
        print("=" * 70)
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n👋 Đã dừng server!")
