import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# ========== 解决中文显示问题 ==========
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 配置参数 ==========
# 数据路径
DATA_DIR = "data"                    # 数据根目录（包含train和val文件夹）
MODEL_SAVE_PATH = "best_model.pth"   # 模型保存路径

# 训练超参数
BATCH_SIZE = 32        # 批次大小（显存不够就改成16或8）
EPOCHS = 20            # 训练轮数
LEARNING_RATE = 0.001  # 学习率
NUM_CLASSES = 6        # 分类数量（TrashNet是6类）

# 设备配置（自动检测GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ========== 数据预处理与增强 ==========
# 训练时的数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # 统一尺寸
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomRotation(15),           # 随机旋转±15度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
    transforms.ToTensor(),                    # 转Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet均值
                         std=[0.229, 0.224, 0.225])    # ImageNet标准差
])

# 验证时只做基本处理（不增强）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_data():
    """加载数据集"""
    print("\n" + "="*60)
    print("加载数据集")
    print("="*60)
    
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'train'), 
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'val'), 
        transform=val_transform
    )
    
    # 打印类别信息
    print(f"\n类别映射: {train_dataset.class_to_idx}")
    print(f"共 {len(train_dataset.classes)} 个类别")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\n训练集图片数: {len(train_dataset)}")
    print(f"验证集图片数: {len(val_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset.classes


def create_model():
    """创建ResNet18模型（迁移学习）"""
    print("\n" + "="*60)
    print("创建模型")
    print("="*60)
    
    # 加载预训练模型
    model = models.resnet18(pretrained=True)
    
    # 冻结前面的层（可选，能加快训练）
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # 替换最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
    
    model = model.to(device)
    
    # 打印模型结构（可选，注释掉避免输出太长）
    # print(model)
    
    print(f"模型已创建，输出类别数: {NUM_CLASSES}")
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [训练]')
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条显示
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='[验证]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'acc': 100.*correct/total})
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def plot_training_history(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1 = axes[0]
    ax1.plot(history['train_loss'], label='训练损失', marker='o', color='blue')
    ax1.plot(history['val_loss'], label='验证损失', marker='s', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练与验证损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2 = axes[1]
    ax2.plot(history['train_acc'], label='训练准确率', marker='o', color='blue')
    ax2.plot(history['val_acc'], label='验证准确率', marker='s', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练与验证准确率曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n📊 训练曲线已保存为: training_history.png")


def print_training_summary(history, best_epoch, best_acc):
    """打印训练总结"""
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    
    print(f"\n最佳模型: Epoch {best_epoch+1}, 验证准确率: {best_acc:.2f}%")
    print(f"最终训练准确率: {history['train_acc'][-1]:.2f}%")
    print(f"最终验证准确率: {history['val_acc'][-1]:.2f}%")
    
    print("\n各Epoch详情:")
    print("-"*60)
    print(f"{'Epoch':<8} {'训练损失':<12} {'训练准确率':<12} {'验证损失':<12} {'验证准确率':<12}")
    print("-"*60)
    for i in range(len(history['train_loss'])):
        print(f"{i+1:<8} {history['train_loss'][i]:<12.4f} {history['train_acc'][i]:<12.2f}% "
              f"{history['val_loss'][i]:<12.4f} {history['val_acc'][i]:<12.2f}%")
    print("="*60)


def main():
    """主训练函数"""
    print("\n" + "="*60)
    print("智能垃圾分类系统 - 模型训练")
    print("="*60)
    
    # 1. 加载数据
    train_loader, val_loader, classes = load_data()
    
    # 2. 创建模型
    model = create_model()
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 可选：学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 4. 训练记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    best_epoch = 0
    
    start_time = time.time()
    
    # 5. 开始训练
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*50}")
        
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印本epoch结果
        print(f"\n📊 Epoch {epoch+1} 结果:")
        print(f"   训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"   验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   ✅ 模型已保存！(准确率 {best_acc:.2f}%)")
        
        # 更新学习率
        scheduler.step()
    
    training_time = time.time() - start_time
    print(f"\n总训练时间: {training_time // 60:.0f}分 {training_time % 60:.0f}秒")
    
    # 6. 绘制训练曲线
    plot_training_history(history)
    
    # 7. 打印总结
    print_training_summary(history, best_epoch, best_acc)
    
    # 8. 保存类别映射（用于推理时）
    class_mapping = {i: classes[i] for i in range(len(classes))}
    import json
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
    print("\n📁 类别映射已保存为: class_mapping.json")
    
    print("\n✅ 训练完成！下一步: 运行 python app.py 启动API服务")


if __name__ == "__main__":
    main()