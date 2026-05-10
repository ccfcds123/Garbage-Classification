from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import torch.nn as nn
from torchvision import models

app = Flask(__name__, static_folder='static')
CORS(app)

# 类别映射（英文 -> 中文 + 垃圾分类）
class_info = {
    'cardboard': {'name': '纸板', 'category': '可回收物', 'advice': '压扁后投入可回收物桶', 'color': '#4CAF50'},
    'glass': {'name': '玻璃', 'category': '可回收物', 'advice': '清洗后投入可回收物桶', 'color': '#4CAF50'},
    'metal': {'name': '金属', 'category': '可回收物', 'advice': '投入可回收物桶', 'color': '#4CAF50'},
    'paper': {'name': '纸张', 'category': '可回收物', 'advice': '叠放整齐后投入可回收物桶', 'color': '#4CAF50'},
    'plastic': {'name': '塑料', 'category': '可回收物', 'advice': '清洗后投入可回收物桶', 'color': '#4CAF50'},
    'trash': {'name': '其他垃圾', 'category': '干垃圾', 'advice': '投入干垃圾桶', 'color': '#795548'}
}

# 加载模型
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 6)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    model = model.to(device)
    return model, device

model, device = load_model()

# 类别映射（索引 -> 英文名）
idx_to_class = {
    0: 'cardboard',
    1: 'glass', 
    2: 'metal',
    3: 'paper',
    4: 'plastic',
    5: 'trash'
}

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取上传的图片
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '没有上传图片'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择图片'})
        
        # 读取并预处理图片
        img = Image.open(file.stream).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # 获取结果
        class_idx = predicted.item()
        class_name = idx_to_class[class_idx]
        info = class_info[class_name]
        
        return jsonify({
            'success': True,
            'prediction': info['name'],
            'category': info['category'],
            'advice': info['advice'],
            'color': info['color']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 智能垃圾分类系统启动中...")
    print("="*50)
    print(f"模型加载完成")
    print(f"访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)