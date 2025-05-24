#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI脑肿瘤分类模型下载器
自动下载训练好的模型文件

使用方法:
    python saved_models/download_models.py
    或
    python download_models.py (在saved_models目录内)
"""

import os
import sys
import hashlib
import urllib.request
import urllib.error
import zipfile
import json
from tqdm import tqdm
import time

class ModelDownloader:
    def __init__(self):
        # 确定脚本所在目录
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = self.script_dir if os.path.basename(self.script_dir) == 'saved_models' else os.path.join(os.getcwd(), 'saved_models')
        
        # 确保目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 模型配置 - 您需要替换为实际的下载链接
        self.model_configs = {
            "resnet18": {
                "filename": "best_resnet18.pth",
                "size_mb": 45,
                "accuracy": "96.33%",
                "description": "ResNet18模型，在Jun Cheng数据集上训练",
                "md5": None,  # 建议添加MD5校验
                "download_urls": [
                    # 主要下载源（替换为您的实际链接）
                    "https://github.com/heyteamo01/mri-brain-tumor-classifier/releases/download/v1.0.0/best_resnet18.pth",
                    # 备用下载源
                    "https://your-cloud-storage.com/models/best_resnet18.pth",
                    # 可以添加更多备用链接
                ]
            },
            "densenet121": {
                "filename": "best_densenet121.pth",
                "size_mb": 28,
                "accuracy": "87.50%", 
                "description": "DenseNet121模型，在Jun Cheng数据集上训练",
                "md5": None,
                "download_urls": [
                    "https://github.com/heyteamo01/mri-brain-tumor-classifier/releases/download/v1.0.0/best_densenet121.pth",
                    "https://your-cloud-storage.com/models/best_densenet121.pth",
                ]
            }
        }
    
    def print_banner(self):
        """显示程序横幅"""
        print("=" * 60)
        print("MRI脑肿瘤分类模型下载器")
        print("=" * 60)
        print("基于Jun Cheng脑肿瘤数据集训练的深度学习模型")
        print("支持ResNet18和DenseNet121两种架构")
        print()
    
    def check_existing_models(self):
        """检查已存在的模型文件"""
        existing_models = []
        missing_models = []
        
        for model_name, config in self.model_configs.items():
            filepath = os.path.join(self.models_dir, config["filename"])
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / 1024 / 1024
                existing_models.append((model_name, config["filename"], size_mb))
            else:
                missing_models.append(model_name)
        
        return existing_models, missing_models
    
    def download_with_progress(self, url, filepath):
        """带进度条的文件下载"""
        try:
            # 获取文件大小
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('Content-Length', 0))
            
            # 下载文件
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=f"下载 {os.path.basename(filepath)}") as pbar:
                
                def update_progress(block_num, block_size, total_size):
                    if pbar.total != total_size and total_size > 0:
                        pbar.total = total_size
                    pbar.update(block_size)
                
                urllib.request.urlretrieve(url, filepath, update_progress)
            
            return True
            
        except urllib.error.URLError as e:
            print(f"下载失败: {e}")
            return False
        except Exception as e:
            print(f"下载过程中出错: {e}")
            return False
    
    def verify_file(self, filepath, expected_md5=None):
        """验证文件完整性"""
        if not os.path.exists(filepath):
            return False
        
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            print(f"警告: {os.path.basename(filepath)} 文件大小为0")
            return False
        
        # MD5校验（如果提供）
        if expected_md5:
            print("验证文件完整性...")
            with open(filepath, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            
            if file_md5.lower() != expected_md5.lower():
                print(f"MD5校验失败: 期望 {expected_md5}, 实际 {file_md5}")
                return False
            print("文件完整性验证通过")
        
        return True
    
    def download_model(self, model_name):
        """下载单个模型"""
        if model_name not in self.model_configs:
            print(f"错误: 未知模型 '{model_name}'")
            return False
        
        config = self.model_configs[model_name]
        filepath = os.path.join(self.models_dir, config["filename"])
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            if self.verify_file(filepath, config["md5"]):
                print(f"模型 {model_name} 已存在且完整，跳过下载")
                return True
            else:
                print(f"模型 {model_name} 文件损坏，重新下载")
                os.remove(filepath)
        
        print(f"\n开始下载 {model_name}:")
        print(f"  描述: {config['description']}")
        print(f"  大小: ~{config['size_mb']}MB")
        print(f"  准确率: {config['accuracy']}")
        
        # 尝试多个下载源
        for i, url in enumerate(config["download_urls"]):
            print(f"\n尝试下载源 {i+1}/{len(config['download_urls'])}...")
            
            if self.download_with_progress(url, filepath):
                if self.verify_file(filepath, config["md5"]):
                    print(f"✅ {model_name} 下载成功")
                    return True
                else:
                    print("文件验证失败，尝试下一个下载源...")
                    if os.path.exists(filepath):
                        os.remove(filepath)
            else:
                print("下载失败，尝试下一个下载源...")
                time.sleep(1)  # 短暂延迟
        
        print(f"❌ {model_name} 下载失败，所有下载源都不可用")
        return False
    
    def download_all_models(self):
        """下载所有模型"""
        existing_models, missing_models = self.check_existing_models()
        
        # 显示现有模型
        if existing_models:
            print("已存在的模型:")
            for name, filename, size in existing_models:
                print(f"  ✅ {name}: {filename} ({size:.1f}MB)")
            print()
        
        # 下载缺失的模型
        if not missing_models:
            print("所有模型文件都已存在，无需下载")
            return True
        
        print("需要下载的模型:")
        for name in missing_models:
            config = self.model_configs[name]
            print(f"  📥 {name}: {config['filename']} (~{config['size_mb']}MB)")
        
        print(f"\n开始下载 {len(missing_models)} 个模型文件...")
        
        success_count = 0
        for model_name in missing_models:
            if self.download_model(model_name):
                success_count += 1
        
        # 结果总结
        print("\n" + "=" * 60)
        print("下载完成!")
        print(f"成功: {success_count}/{len(missing_models)} 个模型")
        
        if success_count == len(missing_models):
            print("🎉 所有模型下载成功！现在可以运行分类器了")
            print("\n运行命令:")
            print("  python mri_classifier.py")
        else:
            print("⚠️ 部分模型下载失败")
            self.show_manual_download_guide()
        
        return success_count == len(missing_models)
    
    def show_manual_download_guide(self):
        """显示手动下载指南"""
        print("\n" + "=" * 60)
        print("手动下载指南")
        print("=" * 60)
        print("如果自动下载失败，您可以手动下载模型文件:")
        print()
        
        print("方法1: GitHub Releases")
        print("  1. 访问: https://github.com/heyteamo01/mri-brain-tumor-classifier/releases")
        print("  2. 下载最新版本的模型文件")
        print("  3. 将文件放入 saved_models/ 目录")
        print()
        
        print("方法2: 云盘下载")
        print("  - 百度网盘: [链接] 提取码: [码]")
        print("  - 阿里云盘: [链接]")
        print("  - Google Drive: [链接]")
        print()
        
        print("方法3: 联系作者")
        print("  - Email: 3240101427@zju.edu.cn")
        print("  - GitHub: https://github.com/heyteamo01")
        print()
        
        print("需要的文件:")
        for config in self.model_configs.values():
            print(f"  - {config['filename']} (~{config['size_mb']}MB)")
    
    def interactive_mode(self):
        """交互式选择模式"""
        existing_models, missing_models = self.check_existing_models()
        
        if not missing_models:
            print("所有模型文件都已存在!")
            return True
        
        print("缺失的模型:")
        for i, name in enumerate(missing_models, 1):
            config = self.model_configs[name]
            print(f"  {i}. {name} - {config['description']} (~{config['size_mb']}MB)")
        
        print(f"  {len(missing_models)+1}. 下载所有模型")
        print("  0. 退出")
        
        while True:
            try:
                choice = input(f"\n请选择要下载的模型 (0-{len(missing_models)+1}): ").strip()
                
                if choice == '0':
                    print("退出下载器")
                    return False
                elif choice == str(len(missing_models)+1):
                    return self.download_all_models()
                else:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(missing_models):
                        model_name = missing_models[choice_idx]
                        return self.download_model(model_name)
                    else:
                        print("无效选择，请重试")
            except (ValueError, KeyboardInterrupt):
                print("\n退出下载器")
                return False

def main():
    """主函数"""
    downloader = ModelDownloader()
    downloader.print_banner()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("使用方法:")
            print("  python download_models.py          # 交互式模式")
            print("  python download_models.py --all    # 下载所有模型")
            print("  python download_models.py resnet18 # 下载指定模型")
            return
        elif sys.argv[1] == '--all':
            downloader.download_all_models()
        elif sys.argv[1] in downloader.model_configs:
            downloader.download_model(sys.argv[1])
        else:
            print(f"错误: 未知参数 '{sys.argv[1]}'")
            print("使用 --help 查看帮助")
    else:
        # 交互式模式
        downloader.interactive_mode()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
