#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI脑肿瘤分类推理系统
基于Jun Cheng脑肿瘤数据集 (https://github.com/chengjun583/brainTumorRetrieval)
支持GPU、Windows、Mac、Linux等多平台
只进行推理，无需训练过程

数据集信息:
- 3064张T1加权对比增强MRI图像
- 233名患者，3种肿瘤类型
- 脑膜瘤(708张)、胶质瘤(1426张)、垂体瘤(930张)
"""

import os
import sys
import platform
import socket
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

# 中文字体设置
plt.rcParams['axes.unicode_minus'] = False
try:
    if platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Microsoft YaHei'
    elif platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'PingFang SC'
    else:  # Linux
        plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    pass

class MRIClassifier:
    """MRI脑肿瘤分类器"""
    
    def __init__(self, model_dir="saved_models"):
        """
        初始化分类器
        
        Args:
            model_dir: 模型文件目录
        """
        self.model_dir = model_dir
        self.device = self._get_device()
        self.models = {}
        self.tumor_types = {
            0: "脑膜瘤 (Meningioma)",     # 708张图像，23.1%
            1: "胶质瘤 (Glioma)",         # 1426张图像，46.5%
            2: "垂体瘤 (Pituitary)"       # 930张图像，30.4%
        }
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        self._load_models()
    
    def _get_device(self):
        """获取最佳计算设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"使用GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')  # Apple Silicon Mac
            print("使用Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            print("使用CPU (建议使用GPU以获得更好性能)")
        
        return device
    
    def _initialize_model(self, model_name, num_classes=3, dropout_rate=0.3):
        """初始化模型结构"""
        if model_name == 'resnet18':
            try:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            except:
                model = models.resnet18(pretrained=True)
            
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
            
        elif model_name == 'densenet121':
            try:
                model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            except:
                model = models.densenet121(pretrained=True)
            
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        return model
    
    def _load_models(self):
        """加载训练好的模型"""
        model_names = ['resnet18', 'densenet121']
        
        for model_name in model_names:
            model_path = os.path.join(self.model_dir, f'best_{model_name}.pth')
            
            if not os.path.exists(model_path):
                print(f"未找到模型文件: {model_path}")
                continue
            
            try:
                # 初始化模型结构
                model = self._initialize_model(model_name).to(self.device)
                
                # 加载模型权重
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if 'best_acc' in checkpoint:
                        print(f"{model_name} 加载成功 (验证准确率: {checkpoint['best_acc']:.4f})")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"{model_name} 加载成功")
                
                model.eval()
                self.models[model_name] = model
                
            except Exception as e:
                print(f"加载 {model_name} 失败: {e}")
        
        if not self.models:
            raise RuntimeError("没有可用的模型！请确保模型文件存在且格式正确。")
        
        print(f"可用模型: {list(self.models.keys())}")
    
    def predict(self, image, model_name='resnet18'):
        """
        预测单张图像
        
        Args:
            image: PIL图像或numpy数组
            model_name: 使用的模型名称
            
        Returns:
            dict: 包含预测结果的字典
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不可用。可用模型: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # 预处理图像
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为tensor
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        # 转换为numpy以便后续处理
        probs_np = probabilities.cpu().numpy()
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': self.tumor_types[predicted_class],
            'confidence': float(probs_np[predicted_class]),
            'all_probabilities': {
                self.tumor_types[i]: float(probs_np[i]) 
                for i in range(len(self.tumor_types))
            }
        }
    
    def predict_batch(self, images, model_name='resnet18'):
        """批量预测"""
        results = []
        for image in images:
            result = self.predict(image, model_name)
            results.append(result)
        return results
    
    def visualize_prediction(self, image, result):
        """可视化预测结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 显示原图
        ax1.imshow(image)
        ax1.set_title('输入图像')
        ax1.axis('off')
        
        # 显示预测概率
        classes = list(result['all_probabilities'].keys())
        probs = list(result['all_probabilities'].values())
        
        bars = ax2.barh(classes, probs)
        ax2.set_xlabel('概率')
        ax2.set_title('预测概率分布')
        ax2.set_xlim(0, 1)
        
        # 突出显示最高概率
        max_idx = probs.index(max(probs))
        bars[max_idx].set_color('red')
        
        # 添加概率标签
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 0.01, i, f'{prob:.3f}', 
                    va='center', fontweight='bold' if i == max_idx else 'normal')
        
        plt.tight_layout()
        return fig

def create_gradio_interface(classifier):
    """创建Gradio界面"""
    
    def predict_interface(image, model_choice):
        """Gradio预测接口"""
        if image is None:
            return None, "请上传MRI图像"
        
        try:
            # 预测
            result = classifier.predict(image, model_choice)
            
            # 生成可视化图表
            fig = classifier.visualize_prediction(image, result)
            
            # 生成文本结果
            result_text = f"""预测结果: {result['predicted_label']}
置信度: {result['confidence']:.4f}

详细概率分布:
"""
            for label, prob in result['all_probabilities'].items():
                result_text += f"  {label}: {prob:.4f}\n"
            
            return fig, result_text
            
        except Exception as e:
            return None, f"预测失败: {str(e)}"
    
    with gr.Blocks(
        title="MRI脑肿瘤分类系统", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {font-family: 'Microsoft YaHei', sans-serif;}
        .main-header {text-align: center; margin-bottom: 2rem;}
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>MRI脑肿瘤分类系统</h1>
            <p>基于深圳大学Jun Cheng数据集的智能诊断助手</p>
            <p><small>数据集: 3064张T1加权对比增强MRI图像，3种肿瘤类型</small></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 输入区域")
                
                image_input = gr.Image(
                    label="上传MRI图像", 
                    type="pil",
                    height=300
                )
                
                model_choice = gr.Radio(
                    choices=list(classifier.models.keys()),
                    label="选择模型",
                    value=list(classifier.models.keys())[0] if classifier.models else None
                )
                
                submit_btn = gr.Button("开始分析", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 使用提示
                - 支持常见图像格式 (jpg, png, bmp等)
                - 建议上传T1加权对比增强MRI脑部横截面图像
                - 系统训练基于512×512像素图像，会自动调整尺寸
                - 支持脑膜瘤、胶质瘤、垂体瘤三种类型识别
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("## 分析结果")
                
                output_plot = gr.Plot(label="预测概率分布图")
                output_text = gr.Textbox(
                    label="详细分析结果", 
                    lines=10,
                    max_lines=15
                )
        
        # 绑定事件
        submit_btn.click(
            fn=predict_interface,
            inputs=[image_input, model_choice],
            outputs=[output_plot, output_text]
        )
        
        gr.HTML("""
        <div style="margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
            <h3>肿瘤类型说明</h3>
            <ul>
                <li><strong>脑膜瘤 (Meningioma)</strong>: 通常为良性，生长缓慢，起源于脑膜 (708例，23.1%)</li>
                <li><strong>胶质瘤 (Glioma)</strong>: 起源于胶质细胞，可能良性或恶性 (1426例，46.5%)</li>
                <li><strong>垂体瘤 (Pituitary)</strong>: 生长在垂体的通常为良性肿瘤 (930例，30.4%)</li>
            </ul>
            <hr>
            <h4>数据集信息</h4>
            <p><strong>来源</strong>: Jun Cheng, 深圳大学生物医学工程学院<br>
            <strong>数据</strong>: 3064张T1加权对比增强MRI图像，来自233名患者<br>
            <strong>引用</strong>: Cheng et al. "Enhanced Performance of Brain Tumor Classification..." PloS one (2015)</p>
            <p><em>注意: 本系统仅供研究和教学使用，不能替代专业医学诊断！</em></p>
        </div>
        """)
    
    return demo

def find_free_port():
    """查找空闲端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def main():
    """主函数"""
    print("=" * 60)
    print("MRI脑肿瘤分类推理系统")
    print("=" * 60)
    
    # 系统信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"PyTorch版本: {torch.__version__}")
    
    try:
        # 初始化分类器
        print("\n初始化模型...")
        classifier = MRIClassifier()
        
        # 创建Gradio界面
        print("\n启动Web界面...")
        demo = create_gradio_interface(classifier)
        
        # 查找空闲端口
        port = find_free_port()
        print(f"服务端口: {port}")
        
        # 启动服务
        demo.launch(
            server_port=port,
            share=False,  # 设为True可生成公网链接
            debug=False,
            show_error=True,
            inbrowser=True  # 自动打开浏览器
        )
        
    except KeyboardInterrupt:
        print("\n用户中断，程序退出")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
