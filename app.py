from flask import Flask, render_template, request, jsonify, send_file
import cv2 as cv
import numpy as np
import base64
import io
from PIL import Image
import os

from utils.normalization import normalize
from utils.segmentation import create_segmented_and_variance_images
from utils import orientation
from utils.frequency import ridge_freq
from utils.gabor_filter import gabor_filter
from utils.skeletonize import skeletonize
from utils.crossing_number import calculate_minutiaes
from utils.poincare import calculate_singularities

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def image_to_base64(img):
    """Chuyển ảnh numpy array sang base64 string"""
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    _, buffer = cv.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def process_fingerprint_step_by_step(input_img):
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
    
    # BƯỚC 2: Chuẩn hóa (Normalization)
    normalized_img = normalize(input_img.copy(), float(100), float(100))
    steps.append({
        'step': 2,
        'name': 'Chuẩn hóa (Normalization)',
        'description': 'Loại bỏ nhiễu từ cảm biến và ảnh hưởng từ lực ấn ngón tay. Chuẩn hóa giá trị pixel về mean=100, variance=100',
        'image': image_to_base64(normalized_img),
        'details': 'Mean: 100, Variance: 100'
    })
    
    # BƯỚC 3: Phân đoạn (Segmentation)
    segmented_img, normim, mask = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
    steps.append({
        'step': 3,
        'name': 'Phân đoạn (Segmentation)',
        'description': 'Tách vùng vân tay ra khỏi vùng nền (background) dựa trên phương sai của từng block',
        'image': image_to_base64(segmented_img),
        'details': f'Block size: {block_size}x{block_size}, Threshold: 0.2'
    })
    
    # BƯỚC 4: Định hướng (Orientation)
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
    steps.append({
        'step': 4,
        'name': 'Định hướng (Orientation)',
        'description': 'Tính toán và hiển thị hướng của các đường vân tay tại mỗi block. Các đường màu xanh lá thể hiện hướng vân tay',
        'image': image_to_base64(orientation_img),
        'details': 'Gradient-based orientation estimation'
    })
    
    # BƯỚC 5: Tần số (Frequency)
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    steps.append({
        'step': 5,
        'name': 'Phân tích Tần số (Frequency Analysis)',
        'description': 'Tính tần số của các đường vân (ridge frequency) để xác định khoảng cách giữa các đường vân',
        'image': image_to_base64(normim),  # Hiển thị normim vì freq là ma trận số
        'details': f'Kernel size: 5, Wave length: 5-15 pixels'
    })
    
    # BƯỚC 6: Lọc Gabor (Gabor Filter)
    gabor_img = gabor_filter(normim, angles, freq)
    steps.append({
        'step': 6,
        'name': 'Lọc Gabor (Gabor Filter)',
        'description': 'Áp dụng bộ lọc Gabor để làm nổi bật đường vân và loại bỏ nhiễu. Gabor filter kết hợp thông tin về hướng và tần số',
        'image': image_to_base64(gabor_img),
        'details': 'Direction-sensitive filtering'
    })
    
    # BƯỚC 7: Làm mỏng (Thinning/Skeletonization)
    thin_image = skeletonize(gabor_img)
    steps.append({
        'step': 7,
        'name': 'Làm mỏng (Skeletonization)',
        'description': 'Làm mỏng các đường vân xuống còn 1 pixel để dễ dàng phát hiện điểm đặc trưng (minutiae)',
        'image': image_to_base64(thin_image),
        'details': 'Ridge thinning to 1-pixel width'
    })
    
    # BƯỚC 8: Điểm đặc trưng Minutiae
    minutias_img = calculate_minutiaes(thin_image)
    steps.append({
        'step': 8,
        'name': 'Điểm đặc trưng (Minutiae Detection)',
        'description': 'Phát hiện các điểm đặc trưng: điểm kết thúc (termination - màu đỏ) và điểm phân nhánh (bifurcation - màu xanh lá)',
        'image': image_to_base64(minutias_img),
        'details': 'Crossing number method'
    })
    
    # BƯỚC 9: Điểm kỳ dị (Singularities)
    singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask)
    steps.append({
        'step': 9,
        'name': 'Điểm kỳ dị (Singularities)',
        'description': 'Phát hiện các điểm kỳ dị (core và delta) - những điểm mà hướng vân tay thay đổi đột ngột',
        'image': image_to_base64(singularities_img),
        'details': 'Poincaré index method'
    })
    
    return steps

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được upload'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400
        
        # Đọc ảnh từ file upload
        file_bytes = np.frombuffer(file.read(), np.uint8)
        input_img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
        
        if input_img is None:
            return jsonify({'error': 'Không thể đọc ảnh. Vui lòng upload file ảnh hợp lệ'}), 400
        
        # Xử lý ảnh qua từng bước
        steps = process_fingerprint_step_by_step(input_img)
        
        return jsonify({
            'success': True,
            'steps': steps
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo', methods=['GET'])
def demo():
    """Load một ảnh demo từ dataset"""
    try:
        # Tìm ảnh demo từ dataset
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
            return jsonify({'error': 'Không tìm thấy ảnh demo trong dataset'}), 404
        
        input_img = cv.imread(demo_img_path, cv.IMREAD_GRAYSCALE)
        
        if input_img is None:
            return jsonify({'error': 'Không thể đọc ảnh demo'}), 400
        
        # Xử lý ảnh qua từng bước
        steps = process_fingerprint_step_by_step(input_img)
        
        return jsonify({
            'success': True,
            'steps': steps,
            'demo_path': demo_img_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Tạo thư mục templates nếu chưa có
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("🚀 KHỞI ĐỘNG ỨNG DỤNG XỬ LÝ VÂN TAY")
    print("=" * 60)
    print("📍 Mở trình duyệt và truy cập: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
