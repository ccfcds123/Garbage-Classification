import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import os

# 加载模型
def load_model(model_path="best_model.pth", num_classes=6):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 类别映射
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
classes_cn = ['纸板', '玻璃', '金属', '纸张', '塑料', '其他垃圾']

# ========== 修改这里：直接指定一张已知的 glass 图片路径 ==========
# 先查看 data/val/glass 文件夹里有什么图片
glass_dir = "data/val/glass"
if os.path.exists(glass_dir):
    available_images = os.listdir(glass_dir)
    print(f"glass 文件夹中的图片: {available_images[:5]}")  # 打印前5张
    
    if available_images:
        test_image_path = os.path.join(glass_dir, available_images[0])
        print(f"使用测试图片: {test_image_path}")
    else:
        print("glass 文件夹为空！")
        exit()
else:
    print(f"文件夹不存在: {glass_dir}")
    exit()

# 加载模型并推理
model = load_model()
img = Image.open(test_image_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

print(f"\n实际类别: glass (玻璃)")
print(f"预测结果: {classes[predicted.item()]} ({classes_cn[predicted.item()]})")

if classes[predicted.item()] == 'glass':
    print("✅ 预测正确！")
else:
    print("❌ 预测错误，模型还需要优化")