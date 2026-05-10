import sys
print(f"Python版本: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    print(f"   GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
except:
    print("❌ PyTorch安装失败")

try:
    import torchvision
    print(f"✅ torchvision版本: {torchvision.__version__}")
except:
    print("❌ torchvision安装失败")

try:
    import flask
    print(f"✅ Flask版本: {flask.__version__}")
except:
    print("❌ Flask安装失败")

try:
    import matplotlib
    print(f"✅ matplotlib版本: {matplotlib.__version__}")
except:
    print("❌ matplotlib安装失败")

try:
    import sklearn
    print(f"✅ scikit-learn版本: {sklearn.__version__}")
except:
    print("❌ scikit-learn安装失败")

try:
    import PIL
    print(f"✅ Pillow版本: {PIL.__version__}")
except:
    print("❌ Pillow安装失败")

print("\n环境检查完成！")