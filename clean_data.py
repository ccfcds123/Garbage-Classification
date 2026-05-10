import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ========== 解决中文显示问题 ==========
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== 配置路径 ==========
RAW_DATA_DIR = r"C:\\Users\\cc\\Desktop\\garbage_classification\\trashnet-master\\trashnet-master\\data\\dataset-resized\\dataset-resized"      # 原始下载的数据集
CLEANED_DATA_DIR = "trashnet_cleaned" # 清洗后的输出路径

# TrashNet的6个类别
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 清洗规则配置
MIN_SIZE = 50        # 最小尺寸阈值（小于50x50的删除）
REQUIRED_MODE = 'RGB' # 要求的颜色模式


def clean_dataset():
    """清洗数据集，返回统计信息"""
    
    # 统计字典
    stats = {}
    
    # 创建清洗后的文件夹
    for class_name in CLASSES:
        os.makedirs(os.path.join(CLEANED_DATA_DIR, class_name), exist_ok=True)
        stats[class_name] = {'total': 0, 'kept': 0, 'deleted': 0, 
                              'deleted_reasons': {'size': 0, 'mode': 0, 'corrupt': 0}}
    
    # 遍历每个类别
    for class_name in CLASSES:
        class_path = os.path.join(RAW_DATA_DIR, class_name)
        
        if not os.path.exists(class_path):
            print(f"⚠️ 跳过不存在的文件夹: {class_path}")
            continue
        
        # 获取所有图片
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        stats[class_name]['total'] = len(images)
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            keep = True
            reason = None
            
            try:
                # 尝试打开图片
                img = Image.open(img_path)
                
                # 检查颜色模式
                if img.mode != REQUIRED_MODE:
                    keep = False
                    reason = 'mode'
                    stats[class_name]['deleted_reasons']['mode'] += 1
                
                # 检查尺寸（只有通过模式检查才检查尺寸）
                if keep and (img.size[0] < MIN_SIZE or img.size[1] < MIN_SIZE):
                    keep = False
                    reason = 'size'
                    stats[class_name]['deleted_reasons']['size'] += 1
                    
            except Exception as e:
                keep = False
                reason = 'corrupt'
                stats[class_name]['deleted_reasons']['corrupt'] += 1
            
            if keep:
                # 复制到清洗后文件夹
                dst_path = os.path.join(CLEANED_DATA_DIR, class_name, img_name)
                shutil.copy2(img_path, dst_path)
                stats[class_name]['kept'] += 1
            else:
                stats[class_name]['deleted'] += 1
                if reason and reason != 'corrupt':
                    print(f"  删除 [{class_name}] {img_name}: {reason}问题")
                elif reason == 'corrupt':
                    print(f"  删除 [{class_name}] {img_name}: 文件损坏")
    
    return stats


def print_stats(stats):
    """打印统计信息"""
    print("\n" + "="*60)
    print("数据清洗统计报告")
    print("="*60)
    
    total_original = 0
    total_kept = 0
    total_deleted = 0
    
    for class_name in CLASSES:
        s = stats[class_name]
        total_original += s['total']
        total_kept += s['kept']
        total_deleted += s['deleted']
        
        print(f"\n📁 {class_name.upper()}")
        print(f"   原始图片: {s['total']} 张")
        print(f"   保留: {s['kept']} 张")
        print(f"   删除: {s['deleted']} 张")
        if s['deleted'] > 0:
            print(f"   删除原因: 尺寸问题 {s['deleted_reasons']['size']} | "
                  f"颜色模式 {s['deleted_reasons']['mode']} | "
                  f"文件损坏 {s['deleted_reasons']['corrupt']}")
    
    print("\n" + "-"*60)
    print(f"📊 总计:")
    print(f"   原始图片总数: {total_original} 张")
    print(f"   保留图片总数: {total_kept} 张")
    print(f"   删除图片总数: {total_deleted} 张")
    print(f"   保留率: {total_kept/total_original*100:.2f}%")
    print("="*60)


def visualize_stats(stats):
    """可视化统计结果"""
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('数据清洗可视化报告', fontsize=16, fontweight='bold')
    
    # 图1：各类别原始 vs 保留（柱状图）
    ax1 = axes[0, 0]
    classes = CLASSES
    original_counts = [stats[c]['total'] for c in classes]
    kept_counts = [stats[c]['kept'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_counts, width, label='原始图片', color='skyblue')
    bars2 = ax1.bar(x + width/2, kept_counts, width, label='清洗后保留', color='lightgreen')
    
    ax1.set_xlabel('类别')
    ax1.set_ylabel('图片数量')
    ax1.set_title('各类别图片数量对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend()
    
    # 在柱子上显示数值
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)
    
    # 图2：各类别删除原因堆叠（水平条形图）
    ax2 = axes[0, 1]
    size_deleted = [stats[c]['deleted_reasons']['size'] for c in classes]
    mode_deleted = [stats[c]['deleted_reasons']['mode'] for c in classes]
    corrupt_deleted = [stats[c]['deleted_reasons']['corrupt'] for c in classes]
    
    ax2.barh(classes, size_deleted, label='尺寸问题', color='salmon')
    ax2.barh(classes, mode_deleted, left=size_deleted, label='颜色模式问题', color='orange')
    ax2.barh(classes, corrupt_deleted, 
             left=[size_deleted[i]+mode_deleted[i] for i in range(len(classes))], 
             label='文件损坏', color='gray')
    
    ax2.set_xlabel('删除数量')
    ax2.set_title('各类别删除原因分析')
    ax2.legend()
    
    # 图3：总体占比饼图
    ax3 = axes[1, 0]
    total_original = sum(stats[c]['total'] for c in classes)
    total_kept = sum(stats[c]['kept'] for c in classes)
    total_deleted = sum(stats[c]['deleted'] for c in classes)
    
    labels = ['保留图片', '删除图片']
    sizes = [total_kept, total_deleted]
    colors = ['lightgreen', 'lightcoral']
    explode = (0, 0.05)
    
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax3.set_title(f'总体保留率: {total_kept/total_original*100:.2f}%')
    
    # 图4：删除原因总体分布
    ax4 = axes[1, 1]
    total_size = sum(stats[c]['deleted_reasons']['size'] for c in classes)
    total_mode = sum(stats[c]['deleted_reasons']['mode'] for c in classes)
    total_corrupt = sum(stats[c]['deleted_reasons']['corrupt'] for c in classes)
    
    reasons = ['尺寸问题', '颜色模式问题', '文件损坏']
    reason_counts = [total_size, total_mode, total_corrupt]
    reason_colors = ['salmon', 'orange', 'gray']
    
    if sum(reason_counts) > 0:
        ax4.pie(reason_counts, labels=reasons, colors=reason_colors,
                autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'删除原因分析 (总计删除{total_deleted}张)')
    else:
        ax4.text(0.5, 0.5, '无删除图片', ha='center', va='center', fontsize=14)
        ax4.set_title('删除原因分析')
    
    plt.tight_layout()
    plt.savefig('data_cleaning_report.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n📊 可视化图表已保存为: data_cleaning_report.png")


def show_sample_comparison():
    """展示清洗前后的图片对比（随机选几张）"""
    
    # 找一个有图片的类别
    for class_name in CLASSES:
        raw_class = os.path.join(RAW_DATA_DIR, class_name)
        cleaned_class = os.path.join(CLEANED_DATA_DIR, class_name)
        
        if os.path.exists(raw_class) and os.path.exists(cleaned_class):
            raw_images = [f for f in os.listdir(raw_class) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            cleaned_images = os.listdir(cleaned_class)
            
            if raw_images and cleaned_images:
                # 随机选3张原始图片和3张清洗后图片
                import random
                sample_raw = random.sample(raw_images, min(3, len(raw_images)))
                sample_cleaned = random.sample(cleaned_images, min(3, len(cleaned_images)))
                
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                fig.suptitle(f'{class_name} 类别 - 清洗前后对比', fontsize=14, fontweight='bold')
                
                for i, img_name in enumerate(sample_raw):
                    img = Image.open(os.path.join(raw_class, img_name))
                    axes[0, i].imshow(img)
                    axes[0, i].set_title(f'原始 {img_name[:15]}...')
                    axes[0, i].axis('off')
                
                for i, img_name in enumerate(sample_cleaned):
                    img = Image.open(os.path.join(cleaned_class, img_name))
                    axes[1, i].imshow(img)
                    axes[1, i].set_title(f'清洗后 {img_name[:15]}...')
                    axes[1, i].axis('off')
                
                plt.tight_layout()
                plt.savefig('sample_comparison.png', dpi=150, bbox_inches='tight')
                plt.show()
                print(f"\n📸 样本对比图已保存为: sample_comparison.png")
                break
            break


if __name__ == "__main__":
    print("开始清洗数据...")
    print(f"原始数据路径: {RAW_DATA_DIR}")
    print(f"清洗后路径: {CLEANED_DATA_DIR}")
    print("-" * 50)
    
    # 执行清洗
    stats = clean_dataset()
    
    # 打印统计信息
    print_stats(stats)
    
    # 可视化统计图表
    visualize_stats(stats)
    
    # 展示清洗前后对比
    show_sample_comparison()
    
    print("\n✅ 数据清洗完成！")