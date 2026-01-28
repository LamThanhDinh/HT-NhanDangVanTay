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
    """Xử lý ảnh vân tay và trả về từng bước"""
    block_size = 16
    steps = []
    
    # BƯỚC 1: Ảnh gốc
    steps.append({
        'step': 1,
        'name': 'Ảnh Gốc (Original Image)',
        'description': 'Ảnh vân tay đầu vào được đọc từ file và chuyển sang ảnh xám (grayscale)',
        'image': image_to_base64(input_img),
        'details': f'Kích thước: {input_img.shape[1]}x{input_img.shape[0]} pixels'
    })
    
    # BƯỚC 2: Chuẩn hóa
    normalized_img = normalize(input_img.copy(), float(100), float(100))
    steps.append({
        'step': 2,
        'name': 'Chuẩn hóa (Normalization)',
        'description': 'Loại bỏ nhiễu từ cảm biến và ảnh hưởng từ lực ấn ngón tay. Chuẩn hóa giá trị pixel về mean=100, variance=100',
        'image': image_to_base64(normalized_img),
        'details': 'Mean: 100, Variance: 100'
    })
    
    # BƯỚC 3: Phân đoạn
    segmented_img, normim, mask = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
    steps.append({
        'step': 3,
        'name': 'Phân đoạn (Segmentation)',
        'description': 'Tách vùng vân tay ra khỏi vùng nền (background) dựa trên phương sai của từng block',
        'image': image_to_base64(segmented_img),
        'details': f'Block size: {block_size}x{block_size}, Threshold: 0.2'
    })
    
    # BƯỚC 4: Định hướng
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
    steps.append({
        'step': 4,
        'name': 'Định hướng (Orientation)',
        'description': 'Tính toán và hiển thị hướng của các đường vân tay tại mỗi block. Các đường màu xanh lá thể hiện hướng vân tay',
        'image': image_to_base64(orientation_img),
        'details': 'Gradient-based orientation estimation'
    })
    
    # BƯỚC 5: Tần số
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    steps.append({
        'step': 5,
        'name': 'Phân tích Tần số (Frequency Analysis)',
        'description': 'Tính tần số của các đường vân (ridge frequency) để xác định khoảng cách giữa các đường vân',
        'image': image_to_base64(normim),
        'details': f'Kernel size: 5, Wave length: 5-15 pixels'
    })
    
    # BƯỚC 6: Lọc Gabor
    gabor_img = gabor_filter(normim, angles, freq)
    steps.append({
        'step': 6,
        'name': 'Lọc Gabor (Gabor Filter)',
        'description': 'Áp dụng bộ lọc Gabor để làm nổi bật đường vân và loại bỏ nhiễu. Gabor filter kết hợp thông tin về hướng và tần số',
        'image': image_to_base64(gabor_img),
        'details': 'Direction-sensitive filtering'
    })
    
    # BƯỚC 7: Làm mỏng
    thin_image = skeletonize(gabor_img)
    steps.append({
        'step': 7,
        'name': 'Làm mỏng (Skeletonization)',
        'description': 'Làm mỏng các đường vân xuống còn 1 pixel để dễ dàng phát hiện điểm đặc trưng (minutiae)',
        'image': image_to_base64(thin_image),
        'details': 'Ridge thinning to 1-pixel width'
    })
    
    # BƯỚC 8: Điểm đặc trưng
    minutias_img = calculate_minutiaes(thin_image)
    steps.append({
        'step': 8,
        'name': 'Điểm đặc trưng (Minutiae Detection)',
        'description': 'Phát hiện các điểm đặc trưng: điểm kết thúc (termination - màu đỏ) và điểm phân nhánh (bifurcation - màu xanh lá)',
        'image': image_to_base64(minutias_img),
        'details': 'Crossing number method'
    })
    
    # BƯỚC 9: Điểm kỳ dị
    singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)
    steps.append({
        'step': 9,
        'name': 'Điểm kỳ dị (Singularities)',
        'description': 'Phát hiện các điểm kỳ dị (core và delta) - những điểm mà hướng vân tay thay đổi đột ngột',
        'image': image_to_base64(singularities_img),
        'details': 'Poincaré index method'
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
