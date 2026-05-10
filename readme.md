# ♻️ 智能垃圾分类回收系统

基于深度学习的智能垃圾分类系统，使用 ResNet18 模型对垃圾图片进行分类，并提供投放建议。

## 📋 项目简介

本项目实现了一个完整的垃圾分类解决方案，包括：

- **数据预处理**：数据清洗、增强、训练/验证集划分
- **模型训练**：基于 ResNet18 迁移学习的图像分类模型
- **Web应用**：Flask后端 + 前端页面的完整演示系统
- **可视化报告**：自动生成数据清洗、划分、训练曲线图表

### 支持的垃圾类别

| 类别 | 垃圾分类 | 投放建议 |
|------|----------|----------|
| 纸板 | 可回收物 | 压扁后投入可回收物桶 |
| 玻璃 | 可回收物 | 清洗后投入可回收物桶 |
| 金属 | 可回收物 | 投入可回收物桶 |
| 纸张 | 可回收物 | 叠放整齐后投入可回收物桶 |
| 塑料 | 可回收物 | 清洗后投入可回收物桶 |
| 其他垃圾 | 干垃圾 | 投入干垃圾桶 |

## 🛠️ 技术栈

- **Python** 3.11+
- **PyTorch** 2.0+ - 深度学习框架
- **Flask** - Web后端框架
- **Matplotlib** - 数据可视化
- **scikit-learn** - 数据集划分

## 📁 项目结构
```

garbage_classification/
│
├── venv/                      # 虚拟环境
├── data/                      # 数据集（训练/验证）
├── trashnet-master/           # 原始TrashNet数据集
├── trashnet_cleaned/          # 清洗后的数据
│
├── clean_data.py              # 数据清洗脚本
├── split_data.py              # 数据划分脚本
├── train.py                   # 模型训练脚本
├── app.py                     # Flask后端API
├── test_model.py              # 模型测试脚本
├── eval_model.py              # 模型评估脚本
├── index.html                 # 前端页面
├── requirements.txt           # 依赖列表
├── check_env.py               # 环境检查脚本
│
├── best_model.pth             # 训练好的模型权重
├── class_mapping.json         # 类别映射文件
├── training_history.png       # 训练曲线图
├── data_cleaning_report.png   # 清洗报告图
├── data_split_report.png      # 划分报告图
│
└── README.md                  # 项目说明

```
## 🚀 快速开始

### 1. 克隆或下载项目

```bash
git clone https://github.com/ccfcds123/Garbage-Classification.git
cd Garbage-Classification
```

### 2. 创建虚拟环境

```bash
python -m venv venv
```

### 3. 激活虚拟环境

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5. 下载数据集

从 [TrashNet](https://github.com/garythung/trashnet) 下载数据集，解压后得到 `trashnet-master/` 文件夹。

### 6. 数据预处理

```bash
# 数据清洗
python clean_data.py

# 划分训练/验证集
python split_data.py
```

### 7. 训练模型

```bash
python train.py
```

训练完成后会生成：

- `best_model.pth` - 最佳模型权重
- `training_history.png` - 训练曲线图

### 8. 测试模型

```bash
python test_model.py
```

### 9. 启动Web应用

```bash
python app.py
```

打开浏览器访问：**[http://localhost:5000](http://localhost:5000)**

## 📊 脚本功能说明

| 脚本 | 功能 | 输入 | 输出 |
| --- | --- | --- | --- |
| `clean_data.py` | 清洗损坏/异常图片 | `trashnet-master/` | `trashnet_cleaned/` + 可视化报告 |
| `split_data.py` | 划分训练/验证集 | `trashnet_cleaned/` | `data/train/`, `data/val/` + 可视化 |
| `train.py` | 训练ResNet18模型 | `data/` | `best_model.pth` + 训练曲线 |
| `test_model.py` | 测试单张图片 | `best_model.pth` | 预测结果 |
| `eval_model.py` | 评估模型准确率 | `best_model.pth` + `data/val/` | 各类别准确率 |
| `app.py` | 启动API服务 | `best_model.pth` | [http://localhost:5000](http://localhost:5000) |
| `check_env.py` | 检查环境配置 | - | 依赖安装状态 |

## 🔧 配置参数

可在 `train.py` 中修改以下参数：

```python
BATCH_SIZE = 32        # 批次大小（显存不足可改为16或8）
EPOCHS = 20            # 训练轮数
LEARNING_RATE = 0.001  # 学习率
TRAIN_RATIO = 0.8      # 训练集比例
```

## 📈 模型性能

| 指标 | 数值 |
| --- | --- |
| 模型 | ResNet18 (迁移学习) |
| 输入尺寸 | 224×224×3 |
| 输出类别 | 6类 |
| 训练集大小 | ~2133张 |
| 验证集大小 | ~533张 |
| 整体准确率 | ~95% (取决于训练效果) |

## 🎯 使用示例

### API调用示例

```python
import requests

url = "http://localhost:5000/predict"
files = {"image": open("test.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### 返回示例

```json
{
    "success": true,
    "prediction": "玻璃",
    "category": "可回收物",
    "advice": "清洗后投入可回收物桶",
    "color": "#4CAF50"
}
```

## 🐛 常见问题

### Q1: 安装依赖时速度太慢

使用清华镜像源：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 虚拟环境无法激活 (Windows)

以管理员身份运行PowerShell，执行：

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q3: 端口5000被占用

修改 `app.py` 最后一行：

```python
app.run(host='0.0.0.0', port=5001, debug=False)
```

然后访问 `http://localhost:5001`

### Q4: 中文图表显示为方框

已在代码中配置中文字体，如仍有问题请安装中文字体或更新matplotlib。

### Q5: CUDA不可用（无法使用GPU）

安装GPU版本的PyTorch：

```bash
# 查看CUDA版本后选择对应命令
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 📄 License

MIT License

## 🙏 致谢

- [TrashNet Dataset](https://github.com/garythung/trashnet) - 垃圾分类数据集
- [PyTorch](https://pytorch.org/) - 深度学习框架
