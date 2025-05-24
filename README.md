# MRI脑肿瘤分类推理系统

基于深度学习的MRI脑肿瘤智能诊断系统，支持脑膜瘤、胶质瘤、垂体瘤三种类型的自动识别。

## 项目概述

本项目是一个完整的MRI脑肿瘤分类推理系统，使用经典的脑肿瘤数据集进行训练，支持多平台部署，提供Web界面和编程API两种使用方式。

### 技术特点
- **免训练使用**：直接加载预训练模型进行推理
- **多平台支持**：Windows、macOS、Linux全兼容
- **GPU加速**：自动检测CUDA、MPS、CPU最佳设备
- **双模型支持**：ResNet18和DenseNet121两种架构
- **友好界面**：基于Gradio的Web界面
- **高精度**：验证准确率可达96%+

## 数据集信息

### 原始数据集
- **来源**：[Jun Cheng - 深圳大学生物医学工程学院](https://github.com/chengjun583/brainTumorRetrieval)
- **数据规模**：3064张T1加权对比增强MRI图像
- **患者数量**：233名患者
- **图像尺寸**：512×512像素
- **许可证**：CC BY 4.0
- **发布地址**：https://doi.org/10.6084/m9.figshare.1512427.v5

### 肿瘤类型分布
- **脑膜瘤 (Meningioma)**：708张图像 (23.1%)
- **胶质瘤 (Glioma)**：1426张图像 (46.5%)  
- **垂体瘤 (Pituitary)**：930张图像 (30.4%)

### 数据集特点
- T1加权对比增强序列，注射Gd-DTPA造影剂
- 体素大小：0.49×0.49×6mm，层间距1mm
- 包含肿瘤边界标注和二值掩码
- 采集时间：2005.9-2010.10

### 相关论文
1. Cheng, Jun, et al. "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition." *PloS one* 10.10 (2015).
2. Cheng, Jun, et al. "Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector Representation." *PloS one* 11.6 (2016).

## 系统要求

### 基础要求
- **Python**: 3.7+
- **内存**: 4GB+ (推荐8GB+)
- **存储**: 2GB+ 可用空间
- **处理器**: 支持AVX指令集的CPU

### GPU支持（可选但推荐）
- **NVIDIA GPU**: GTX 1060 / RTX 2060及以上，CUDA 11.0+
- **Apple Silicon**: M1/M2系列，自动支持MPS加速
- **仅CPU**: 也可正常运行，推理时间约慢10倍

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-username/mri-brain-tumor-classifier.git
cd mri-brain-tumor-classifier

# 创建虚拟环境（推荐）
python -m venv mri_env

# 激活虚拟环境
# Windows:
mri_env\Scripts\activate
# macOS/Linux:
source mri_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型准备

确保在 `saved_models/` 目录下有训练好的模型文件：
- `best_resnet18.pth`
- `best_densenet121.pth`

如果没有模型文件，请：
1. 下载预训练模型：[模型下载链接]
2. 或使用提供的训练代码自行训练

### 3. 启动系统

```bash
# 启动Web界面
python mri_classifier.py

# 系统会自动：
# - 检测最佳计算设备
# - 加载训练好的模型  
# - 启动Web服务
# - 打开浏览器
```

### 4. 使用系统

1. 在Web界面上传MRI图像（支持jpg、png、bmp等格式）
2. 选择预测模型（ResNet18或DenseNet121）
3. 点击"开始分析"按钮
4. 查看预测结果和概率分布图

## 编程接口使用

### 基础预测示例

```python
from mri_classifier import MRIClassifier
from PIL import Image

# 初始化分类器
classifier = MRIClassifier(model_dir="saved_models")

# 加载图像
image = Image.open("sample_mri.jpg")

# 单张预测
result = classifier.predict(image, model_name='resnet18')

print(f"预测结果: {result['predicted_label']}")
print(f"置信度: {result['confidence']:.4f}")

# 查看所有概率
for label, prob in result['all_probabilities'].items():
    print(f"{label}: {prob:.4f}")
```

### 批量预测示例

```python
# 批量处理
images = [Image.open(f"test_{i}.jpg") for i in range(1, 6)]
results = classifier.predict_batch(images, model_name='resnet18')

for i, result in enumerate(results):
    print(f"图像{i+1}: {result['predicted_label']} (置信度: {result['confidence']:.3f})")
```

### 结果可视化

```python
import matplotlib.pyplot as plt

# 预测并可视化
result = classifier.predict(image)
fig = classifier.visualize_prediction(image, result)
plt.show()
```

## 项目结构

```
mri_classifier/
├── mri_classifier.py           # 主程序
├── requirements.txt            # 依赖列表
├── README.md                   # 使用说明
├── LICENSE                     # 许可证
└── saved_models/              # 模型文件目录
    ├── best_resnet18.pth      # ResNet18模型
    └── best_densenet121.pth   # DenseNet121模型
```

## 性能基准

### 模型性能
| 模型 | 验证准确率 | 推理时间 | 模型大小 |
|------|-----------|----------|----------|
| ResNet18 | 96.33% | ~50ms | ~45MB |
| DenseNet121 | 87.50% | ~80ms | ~28MB |

### 分类性能（ResNet18）
| 类别 | 精确率 | 召回率 | F1分数 | AUC |
|------|--------|--------|--------|-----|
| 脑膜瘤 | 92% | 95% | 93% | 0.995 |
| 胶质瘤 | 98% | 95% | 97% | 0.997 |
| 垂体瘤 | 97% | 100% | 99% | 0.999 |
| **平均** | **96%** | **97%** | **96%** | **0.997** |

## 配置选项

### config.json 配置文件

```json
{
    "model_dir": "saved_models",
    "default_model": "resnet18",
    "batch_size": 32,
    "confidence_threshold": 0.7,
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "image_preprocessing": {
        "resize": 256,
        "crop_size": 224,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    },
    "gradio_settings": {
        "share": false,
        "debug": false,
        "inbrowser": true,
        "show_error": true
    }
}
```

## 部署选项

### 1. 本地部署
```bash
python mri_classifier.py
```

### 2. Docker部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "mri_classifier.py"]
```

### 3. 云端部署
支持部署到：
- **Hugging Face Spaces**
- **Google Colab**  
- **Azure Container Instances**
- **AWS EC2**

## 常见问题

### Q: 模型文件太大，如何处理？
A: 
1. 使用Git LFS管理大文件
2. 将模型放在云存储，程序启动时下载
3. 使用模型量化减小文件大小

### Q: CUDA内存不足怎么办？
A:
```python
# 强制使用CPU
classifier = MRIClassifier()
classifier.device = torch.device('cpu')
```

### Q: 如何添加新模型？
A: 在 `_initialize_model` 方法中添加新的模型架构，然后训练并保存模型文件。

### Q: 如何提高预测精度？
A:
1. 使用更多数据增强
2. 尝试集成学习（多模型投票）
3. 调整预处理参数
4. 使用更先进的模型架构

## 医学免责声明

**重要提醒**：
- 本系统仅供研究和教学使用
- 不能替代专业医学诊断
- 临床应用必须经过专业医生确认
- 作者不承担任何医疗责任

## 数据使用声明

本项目使用的数据集遵循以下条款：
- **数据来源**：Jun Cheng, 深圳大学
- **许可证**：CC BY 4.0
- **引用要求**：使用数据请引用相关论文
- **商业使用**：需要额外授权

## 许可证

本项目采用 [MIT许可证](LICENSE)。

## 贡献指南

欢迎提交Issue和Pull Request！

### 贡献类型
- 🐛 Bug修复
- ✨ 新功能
- 📝 文档改进
- 🎨 代码优化
- 🧪 测试用例

### 开发环境搭建
```bash
# 克隆开发分支
git clone -b develop https://github.com/your-username/mri-brain-tumor-classifier.git

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
pytest tests/

# 代码格式化
black mri_classifier.py
```

## 联系方式

- **GitHub Issues**: [[Issues]](https://github.com/heyteamo01/MRI-/issues)
- **Email**: 3240101427@zju.edu.cn or heytea01@gmail.com

## 致谢

感谢以下贡献：
- **Jun Cheng教授**：提供原始数据集
- **深圳大学生物医学工程学院**：数据采集和标注
- **开源社区**：PyTorch、Gradio等优秀框架

---

*最后更新：2025年5月25日*
*版本：v1.0.1*
