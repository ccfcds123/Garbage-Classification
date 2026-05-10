import os
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# ========== 解决中文显示问题 ==========
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 配置路径 ==========
SOURCE_DIR = "trashnet_cleaned"   # 清洗后的数据路径
DEST_DIR = "data"                  # 划分后的输出路径

# 类别（英文原路径名）
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# 中文显示名
CLASSES_CN = ['纸板', '玻璃', '金属', '纸张', '塑料', '其他垃圾']

# 划分比例
TRAIN_RATIO = 0.8  # 80%训练，20%验证
RANDOM_SEED = 42   # 随机种子，保证每次划分结果一致


def split_dataset():
    """划分训练集和验证集，返回统计信息"""
    
    stats = {}
    
    for i, class_name in enumerate(CLASSES):
        class_source = os.path.join(SOURCE_DIR, class_name)
        
        if not os.path.exists(class_source):
            print(f"⚠️ 跳过不存在的文件夹: {class_source}")
            continue
        
        # 获取所有图片
        images = [f for f in os.listdir(class_source) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) == 0:
            print(f"⚠️ {class_name} 文件夹中没有图片")
            continue
        
        # 随机划分
        train_imgs, val_imgs = train_test_split(
            images, 
            train_size=TRAIN_RATIO, 
            random_state=RANDOM_SEED,
            shuffle=True
        )
        
        # 创建目标文件夹
        train_dest = os.path.join(DEST_DIR, 'train', class_name)
        val_dest = os.path.join(DEST_DIR, 'val', class_name)
        os.makedirs(train_dest, exist_ok=True)
        os.makedirs(val_dest, exist_ok=True)
        
        # 复制训练集
        for img in train_imgs:
            src = os.path.join(class_source, img)
            dst = os.path.join(train_dest, img)
            shutil.copy2(src, dst)
        
        # 复制验证集
        for img in val_imgs:
            src = os.path.join(class_source, img)
            dst = os.path.join(val_dest, img)
            shutil.copy2(src, dst)
        
        # 记录统计信息
        stats[class_name] = {
            'cn_name': CLASSES_CN[i],
            'total': len(images),
            'train': len(train_imgs),
            'val': len(val_imgs),
            'train_ratio': len(train_imgs) / len(images) * 100,
            'val_ratio': len(val_imgs) / len(images) * 100
        }
        
        print(f"{CLASSES_CN[i]:<6} ({class_name}): 总计 {len(images):>4}张 | "
              f"训练 {len(train_imgs):>4}张 ({len(train_imgs)/len(images)*100:.1f}%) | "
              f"验证 {len(val_imgs):>4}张 ({len(val_imgs)/len(images)*100:.1f}%)")
    
    return stats


def print_summary(stats):
    """打印汇总统计"""
    print("\n" + "="*60)
    print("数据划分汇总报告")
    print("="*60)
    
    total_images = sum(s['total'] for s in stats.values())
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    
    print(f"\n📊 总计:")
    print(f"   总图片数: {total_images} 张")
    print(f"   训练集: {total_train} 张 ({total_train/total_images*100:.1f}%)")
    print(f"   验证集: {total_val} 张 ({total_val/total_images*100:.1f}%)")
    print(f"   划分比例: 训练 {TRAIN_RATIO*100:.0f}% / 验证 {(1-TRAIN_RATIO)*100:.0f}%")
    print("="*60)


def visualize_split(stats):
    """可视化数据划分结果"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('数据集划分可视化报告', fontsize=16, fontweight='bold')
    
    # 图1：各类别训练/验证集数量对比（柱状图）
    ax1 = axes[0, 0]
    categories = [s['cn_name'] for s in stats.values()]
    train_counts = [s['train'] for s in stats.values()]
    val_counts = [s['val'] for s in stats.values()]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_counts, width, label='训练集', color='steelblue')
    bars2 = ax1.bar(x + width/2, val_counts, width, label='验证集', color='coral')
    
    ax1.set_xlabel('类别')
    ax1.set_ylabel('图片数量')
    ax1.set_title('各类别训练/验证集数量对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    
    # 显示数值
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)
    
    # 图2：总体划分比例（饼图）
    ax2 = axes[0, 1]
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    
    labels = [f'训练集\n{total_train}张', f'验证集\n{total_val}张']
    sizes = [total_train, total_val]
    colors = ['steelblue', 'coral']
    explode = (0, 0.05)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title(f'总体划分比例 (训练:{TRAIN_RATIO*100:.0f}% / 验证:{(1-TRAIN_RATIO)*100:.0f}%)')
    
    # 图3：各类别训练集占比（柱状图）
    ax3 = axes[1, 0]
    train_ratios = [s['train_ratio'] for s in stats.values()]
    
    bars3 = ax3.bar(categories, train_ratios, color='steelblue', alpha=0.7)
    ax3.axhline(y=TRAIN_RATIO*100, color='red', linestyle='--', label=f'目标比例 {TRAIN_RATIO*100:.0f}%')
    ax3.set_xlabel('类别')
    ax3.set_ylabel('训练集占比 (%)')
    ax3.set_title('各类别训练集占比')
    ax3.set_ylim(0, 105)
    ax3.legend()
    
    for bar, ratio in zip(bars3, train_ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{ratio:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 图4：各类别数量分布（堆叠面积图）
    ax4 = axes[1, 1]
    class_indices = np.arange(len(categories))
    
    # 绘制堆叠条形图
    ax4.barh(class_indices, train_counts, label='训练集', color='steelblue', alpha=0.8)
    ax4.barh(class_indices, val_counts, left=train_counts, label='验证集', color='coral', alpha=0.8)
    
    ax4.set_yticks(class_indices)
    ax4.set_yticklabels(categories)
    ax4.set_xlabel('图片数量')
    ax4.set_title('各类别数据分布（堆叠图）')
    ax4.legend()
    
    # 显示总数
    for i, (train, val, cn_name) in enumerate(zip(train_counts, val_counts, categories)):
        total = train + val
        ax4.text(total + 2, i, f'总计:{total}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('data_split_report.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n📊 可视化图表已保存为: data_split_report.png")


def verify_split():
    """验证划分结果是否正确"""
    print("\n" + "="*60)
    print("验证划分结果")
    print("="*60)
    
    for class_name in CLASSES:
        train_path = os.path.join(DEST_DIR, 'train', class_name)
        val_path = os.path.join(DEST_DIR, 'val', class_name)
        
        train_count = len(os.listdir(train_path)) if os.path.exists(train_path) else 0
        val_count = len(os.listdir(val_path)) if os.path.exists(val_path) else 0
        
        # 检查是否有文件重复
        if os.path.exists(train_path) and os.path.exists(val_path):
            train_files = set(os.listdir(train_path))
            val_files = set(os.listdir(val_path))
            overlap = train_files & val_files
            
            if overlap:
                print(f"⚠️ {class_name}: 发现 {len(overlap)} 个重复文件!")
            else:
                print(f"✓ {class_name}: 训练集 {train_count}张, 验证集 {val_count}张, 无重复")
    
    print("="*60)


def show_split_example():
    """展示划分后的文件夹结构示例"""
    print("\n📁 划分后的文件夹结构:")
    print(f"{DEST_DIR}/")
    
    for class_name in CLASSES:
        train_path = os.path.join(DEST_DIR, 'train', class_name)
        val_path = os.path.join(DEST_DIR, 'val', class_name)
        
        if os.path.exists(train_path):
            train_sample = os.listdir(train_path)[:3]  # 取前3个文件名
            print(f"├── train/{class_name}/")
            for img in train_sample:
                print(f"│   └── {img}")
            if len(os.listdir(train_path)) > 3:
                print(f"│   └── ... 共{len(os.listdir(train_path))}张")
        
        if os.path.exists(val_path):
            val_sample = os.listdir(val_path)[:3]
            print(f"└── val/{class_name}/")
            for img in val_sample:
                print(f"    └── {img}")
            if len(os.listdir(val_path)) > 3:
                print(f"    └── ... 共{len(os.listdir(val_path))}张")
        print()


if __name__ == "__main__":
    print("开始划分数据集...")
    print(f"源数据路径: {SOURCE_DIR}")
    print(f"目标路径: {DEST_DIR}")
    print(f"划分比例: 训练集 {TRAIN_RATIO*100:.0f}% / 验证集 {(1-TRAIN_RATIO)*100:.0f}%")
    print("-" * 50)
    
    # 执行划分
    stats = split_dataset()
    
    # 打印汇总
    print_summary(stats)
    
    # 可视化
    visualize_split(stats)
    
    # 验证
    verify_split()
    
    # 展示结构
    show_split_example()
    
    print("\n✅ 数据划分完成！")