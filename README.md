# MRI脑肿瘤分类推理系统

## 项目概述

这是一个基于深度学习的MRI脑肿瘤分类推理系统，支持多平台运行（Windows、Mac、Linux），可以自动识别脑膜瘤、胶质瘤、垂体瘤三种肿瘤类型。

## 系统要求

### 基础要求
- Python 3.7+
- 内存: 4GB+ (推荐8GB+)
- 存储: 2GB+ 可用空间

### GPU支持（可选但推荐）
- **NVIDIA GPU**: CUDA 11.0+
- **Apple Silicon Mac**: 自动支持MPS
- **仅CPU**: 也可运行，但速度较慢

## 安装依赖

### 1. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv mri_env
mri_env\Scripts\activate

# macOS/Linux  
python3 -m venv mri_env
source mri_env/bin/activate
```

### 2. 安装依赖包

```bash
# 基础依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 如果有NVIDIA GPU，使用CUDA版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install gradio pillow matplotlib numpy scikit-learn
```

### 3. requirements.txt文件

```txt
torch>=1.12.0
torchvision>=0.13.0
gradio>=3.40.0
pillow>=8.0.0
matplotlib>=3.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

使用方式：
```bash
pip install -r requirements.txt
```

## 项目结构

```
mri_classifier/
├── mri_classifier.py          # 主程序文件
├── usage_example.py           # 使用示例
├── requirements.txt           # 依赖文件
├── config.json               # 配置文件
├── saved_models/             # 模型文件目录
│   ├── best_resnet18.pth
│   └── best_densenet121.pth
└── test_images/              # 测试图像目录
    ├── sample1.jpg
    └── sample2.jpg
```

## 快速开始

### 1. 准备模型文件

确保在 `saved_models/` 目录下有以下文件：
- `best_resnet18.pth`
- `best_densenet121.pth`

### 2. 启动Web界面

```bash
python mri_classifier.py
```

程序会自动：
- 检测最佳计算设备（GPU/CPU）
- 加载训练好的模型
- 启动Web界面
- 打开浏览器

### 3. 使用系统

1. 在Web界面上上传MRI图像
2. 选择预测模型（ResNet18或DenseNet121）
3. 点击"开始分析"按钮
4. 查看预测结果和概率分布

## 编程接口使用

### 基础预测

```python
from mri_classifier import MRIClassifier
from PIL import Image

# 初始化分类器
classifier = MRIClassifier(model_dir="saved_models")

# 加载图像
image = Image.open("test.jpg")

# 预测
result = classifier.predict(image, model_name='resnet18')

print(f"预测结果: {result['predicted_label']}")
print(f"置信度: {result['confidence']:.4f}")
```

### 批量预测

```python
images = [Image.open(f"test{i}.jpg") for i in range(1, 6)]
results = classifier.predict_batch(images, model_name='resnet18')

for i, result in enumerate(results):
    print(f"图像{i+1}: {result['predicted_label']} ({result['confidence']:.3f})")
```

### 结果可视化

```python
import matplotlib.pyplot as plt

# 预测并可视化
result = classifier.predict(image)
fig = classifier.visualize_prediction(image, result)
plt.show()
```

## 配置选项

### config.json 配置文件

```json
{
    "model_dir": "saved_models",
    "default_model": "resnet18",
    "batch_size": 32,
    "confidence_threshold": 0.7,
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "gradio_settings": {
        "share": false,
        "debug": false,
        "inbrowser": true,
        "show_error": true
    }
}
```

## 常见问题

### 1. CUDA内存不足
```python
# 减少批处理大小或使用CPU
classifier = MRIClassifier()
# 强制使用CPU
classifier.device = torch.device('cpu')
```

### 2. 模型文件损坏
```bash
# 重新下载模型文件
# 检查文件完整性
python -c "import torch; torch.load('saved_models/best_resnet18.pth')"
```

### 3. 端口被占用
```python
# 在代码中指定端口
demo.launch(server_port=8080)
```

### 4. 中文字体显示问题
```python
# Windows
plt.rcParams['font.family'] = 'Microsoft YaHei'
# macOS
plt.rcParams['font.family'] = 'PingFang SC'
# Linux 
plt.rcParams['font.family'] = 'DejaVu Sans'
```

## 性能基准

| 平台 | 设备 | 单张预测时间 | 内存使用 |
|------|------|-------------|---------|
| Windows | RTX 3080 | ~50ms | ~2GB |
| macOS | M1 Pro | ~80ms | ~1.5GB |
| Linux | CPU only | ~800ms | ~1GB |

## 安全说明

**重要提醒**: 
- 本系统仅供研究和教学使用
- 不能替代专业医学诊断
- 临床应用需要专业医生确认

## 许可证

本项目采用 MIT 许可证，详情请参阅 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请通过以下方式联系：
- GitHub Issues
- Email: 3240101427@zju.edu.cn

---

*最后更新: 2025年5月25日*
