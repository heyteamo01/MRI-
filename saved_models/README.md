# 模型文件下载说明

由于GitHub文件大小限制，训练好的模型文件需要从外部存储下载。

## 🎯 需要的模型文件

本项目需要以下模型文件：

| 模型文件 | 大小 | 验证准确率 | 描述 |
|---------|------|-----------|------|
| `best_resnet18.pth` | ~45MB | 96.33% | ResNet18架构 |
| `best_densenet121.pth` | ~28MB | 87.50% | DenseNet121架构 |

## 📥 下载方式

### 方式1：自动下载脚本（推荐）

```bash
# 运行下载脚本
python download_models.py

# 或者在主程序中会自动提示下载
python mri_classifier.py
```

### 方式2：手动下载

#### 选项A：云盘下载
- **百度网盘**: [下载链接](https://pan.baidu.com/s/1sl1W9SIA2waZB539k-ivWg) 提取码: `mrt0`
- **Google Drive**: [下载链接](https://drive.google.com/drive/folders/1hXSOIyhXtjlKKjsj5AzvIzUV58TGz0Th?usp=drive_link)

#### 选项B：GitHub Releases
- 访问项目的 [Releases页面](https://github.com/heyteamo01/mri-brain-tumor-classifier/releases/tag/mrt_classifier_model)
- 下载最新版本的模型文件压缩包
- 解压到当前目录

## 📁 文件放置

下载后，请确保文件结构如下：

```
saved_models/
├── README.md                 # 本文件
├── best_resnet18.pth        # ResNet18模型
├── best_densenet121.pth     # DenseNet121模型
└── .gitkeep                 # 保持目录结构
```

## ✅ 验证安装

运行以下命令验证模型文件是否正确：

```bash
python -c "
import os
models = ['best_resnet18.pth', 'best_densenet121.pth']
for model in models:
    path = f'saved_models/{model}'
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024 / 1024
        print(f'✅ {model}: {size:.1f}MB')
    else:
        print(f'❌ {model}: 文件不存在')
"
```

## 🔧 重新训练模型（可选）

如果您有Jun Cheng数据集，也可以重新训练模型：

```bash
# 1. 下载数据集
# https://doi.org/10.6084/m9.figshare.1512427.v5

# 2. 运行训练脚本
python train_model.py

# 3. 训练完成后，模型会自动保存到saved_models/目录
```

## 🔒 文件完整性

为确保下载的文件完整，您可以验证文件哈希：

```python
import hashlib

def verify_model(filepath, expected_md5):
    with open(filepath, 'rb') as f:
        file_md5 = hashlib.md5(f.read()).hexdigest()
    return file_md5 == expected_md5

# 验证示例（请替换为实际的MD5值）
verify_model('saved_models/best_resnet18.pth', 'actual_md5_hash_here')
```

---

**注意**: 模型文件基于Jun Cheng脑肿瘤数据集训练，仅供研究和教学使用。
