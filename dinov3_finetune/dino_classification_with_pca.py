"""
DINOv3 Feature Extraction + Classification + PCA Visualization
用未微调的DINOv3模型提取特征，进行分类（MLP神经网络），并生成PCA可视化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
from pathlib import Path
import argparse
import numpy as np
import json


# ===========================
# 1. 配置参数
# ===========================
def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 Classification with PCA Visualization")
    parser.add_argument("--train_1T_dir", type=str, default="../PCA/train_pic/Sim_1T_256", 
                        help="1T相训练图片目录")
    parser.add_argument("--train_2H_dir", type=str, default="../PCA/train_pic/Sim_2H_256",
                        help="2H相训练图片目录")
    parser.add_argument("--test_1T_dir", type=str, default="../PCA/train_pic/Sim_1T_256",
                        help="1T相测试图片目录")
    parser.add_argument("--test_2H_dir", type=str, default="../PCA/train_pic/Sim_2H_256",
                        help="2H相测试图片目录")
    parser.add_argument("--output_dir", type=str, default="./classification_results",
                        help="输出结果目录")
    parser.add_argument("--weights_path", type=str, 
                        default="../dinov3/weight/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
                        help="DINOv3预训练权重路径")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16plus",
                        help="DINOv3模型名称")
    parser.add_argument("--repo_dir", type=str, default="../dinov3",
                        help="DINOv3仓库路径")
    parser.add_argument("--epochs", type=int, default=20,
                        help="MLP训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="训练批次大小")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--hidden_dims", type=str, default="128,64",
                        help="隐藏层维度，用逗号分隔，例如：128,64")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout比例")
    parser.add_argument("--random_state", type=int, default=42,
                        help="随机种子")
    return parser.parse_args()


# ===========================
# 2. 加载DINOv3模型
# ===========================
def get_dino_model(repo_dir, model_name='dinov3_vits16plus', weights_path=None):
    """加载DINOv3模型"""
    model = torch.hub.load(repo_dir, model_name, source='local', weights=weights_path)
    return model


# ===========================
# 3. 特征提取模块
# ===========================
class FeatureExtractor:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # 预处理：调整到1024大小
        self.transform_resize = transforms.RandomResizedCrop(
            size=(1024, 1024),
            scale=(1.0, 1.0),
            ratio=(1.0, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        
        self.transform_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def extract_features(self, img_path):
        """
        提取单张图片的特征
        返回: cls_token特征 (用于分类), patch_tokens特征 (用于PCA可视化)
        """
        img = Image.open(img_path).convert('RGB')
        
        # 调整到1024x1024
        img = self.transform_resize(img)
        W_orig, H_orig = img.size
        
        # 转换为tensor
        img_tensor = self.transform_tensor(img).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
            cls_token = features['x_norm_clstoken'].squeeze(0).cpu().numpy()  # [emb_dim]
            patch_tokens = features['x_norm_patchtokens'].squeeze(0).cpu().numpy()  # [num_patches, emb_dim]
        
        return cls_token, patch_tokens, img, (H_orig, W_orig)


# ===========================
# 4. PCA可视化模块
# ===========================
class PCAVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_pca_combined(self, img, patch_tokens, img_name, original_size, 
                               predicted_label=None, true_label=None, n_train_samples=None):
        """
        生成组合图：左边原图，右边PCA热图
        img: PIL Image (1024x1024)
        patch_tokens: [num_patches, emb_dim]
        img_name: 图片文件名
        original_size: (H, W) 原始图片尺寸
        predicted_label: 预测标签 (0=1T, 1=2H)
        true_label: 真实标签 (0=1T, 1=2H)
        n_train_samples: 训练样本数
        """
        H_orig, W_orig = original_size
        
        # 计算patch grid大小
        h_grid = 1024 // 16
        w_grid = 1024 // 16
        
        # PCA降维到1个成分用于热图
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(patch_tokens)
        heatmap = pca_result.reshape(h_grid, w_grid)
        
        # 归一化
        p5, p95 = np.percentile(heatmap, [5, 95])
        heatmap = np.clip(heatmap, p5, p95)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # 上采样到原始尺寸
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
        upsampled = torch.nn.functional.interpolate(
            heatmap_tensor,
            size=(H_orig, W_orig),
            mode='bicubic',
            align_corners=False
        )
        heatmap_full = upsampled.squeeze().numpy()
        heatmap_full = np.clip(heatmap_full, 0, 1)
        
        # 创建组合图：左边原图，右边PCA热图
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左边：原图
        axes[0].imshow(img)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis('off')
        
        # 右边：PCA热图
        axes[1].imshow(heatmap_full, cmap='inferno')
        axes[1].set_title("PCA Heatmap", fontsize=14)
        axes[1].axis('off')
        
        # 设置总标题
        if predicted_label is not None:
            pred_name = '1T' if predicted_label == 0 else '2H'
            true_name = '1T' if true_label == 0 else '2H'
            is_correct = predicted_label == true_label
            status = '✓' if is_correct else '✗'
            
            title = f"{status} Prediction: {pred_name} | True Label: {true_name} | File: {img_name}"
            if n_train_samples:
                title += f" | Training Samples: {n_train_samples}"
            
            # 设置颜色：正确=绿色，错误=红色
            color = 'green' if is_correct else 'red'
            fig.suptitle(title, fontsize=16, fontweight='bold', color=color)
        else:
            title = f"Image: {img_name}"
            if n_train_samples:
                title += f" | Training Samples: {n_train_samples}"
            fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存组合图
        combined_path = os.path.join(self.output_dir, f"{Path(img_name).stem}_combined.png")
        plt.savefig(combined_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return combined_path


# ===========================
# 5. 数据加载与特征提取
# ===========================
def load_and_extract_features(data_dir, label, feature_extractor, pca_visualizer=None, 
                              predictions=None, n_train_samples=None):
    """
    加载数据集并提取特征
    data_dir: 数据目录
    label: 类别标签 (0 for 1T, 1 for 2H)
    feature_extractor: FeatureExtractor实例
    pca_visualizer: PCAVisualizer实例 (可选)
    predictions: 预测结果列表 (可选)
    n_train_samples: 训练样本数 (可选)
    """
    features = []
    labels = []
    image_info = []
    
    data_path = Path(data_dir)
    image_files = sorted(list(data_path.glob('*.png')) + list(data_path.glob('*.jpg')) + list(data_path.glob('*.jpeg')))
    
    print(f"Processing {len(image_files)} images from {data_dir}...")
    
    for idx, img_path in enumerate(image_files):
        # 提取特征
        cls_token, patch_tokens, img, original_size = feature_extractor.extract_features(img_path)
        
        features.append(cls_token)
        labels.append(label)
        
        # 保存图片信息
        info = {
            'filename': img_path.name,
            'label': label,
            'label_name': '1T' if label == 0 else '2H'
        }
        
        # 生成PCA可视化组合图
        if pca_visualizer:
            pred_label = predictions[idx] if predictions is not None else None
            combined_path = pca_visualizer.visualize_pca_combined(
                img, patch_tokens, img_path.name, original_size,
                predicted_label=pred_label,
                true_label=label,
                n_train_samples=n_train_samples
            )
            info['combined_path'] = combined_path
        
        image_info.append(info)
        print(f"  Processed: {img_path.name}")
    
    return np.array(features), np.array(labels), image_info


# ===========================
# 6. MLP神经网络分类器
# ===========================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dims=[128, 64], num_classes=2, dropout=0.3):
        """
        简单的MLP分类器
        input_dim: 输入特征维度
        hidden_dims: 隐藏层维度列表
        num_classes: 分类类别数
        dropout: Dropout比例
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPClassifierModel:
    def __init__(self, input_dim=384, hidden_dims=[128, 64], num_classes=2, 
                 dropout=0.3, lr=0.001, device='cuda'):
        """MLP分类器模型封装"""
        self.device = device
        self.model = MLPClassifier(input_dim, hidden_dims, num_classes, dropout).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def train(self, X_train, y_train, epochs=20, batch_size=8):
        """训练MLP模型"""
        print(f"\nTraining MLP with {len(X_train)} samples for {epochs} epochs...")
        
        # 准备数据
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            # 打印训练进度
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100 * correct / total
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        print("Training completed!")
    
    def predict(self, X_test):
        """预测"""
        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=['1T', '2H'])
        
        print("\n" + "="*50)
        print("Classification Results")
        print("="*50)
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"\nConfusion Matrix:\n{conf_matrix}")
        print(f"\nClassification Report:\n{class_report}")
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist()
        }


# ===========================
# 7. 结果保存模块
# ===========================
def save_results(output_dir, test_info, predictions, evaluation_results):
    """
    保存所有结果
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 更新测试信息，添加预测结果
    for i, info in enumerate(test_info):
        info['predicted_label'] = int(predictions[i])
        info['predicted_name'] = '1T' if predictions[i] == 0 else '2H'
        info['correct'] = (info['label'] == predictions[i])
    
    # 保存详细结果为JSON
    results_json = {
        'evaluation': evaluation_results,
        'test_images': test_info
    }
    
    json_path = os.path.join(output_dir, 'classification_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {json_path}")
    
    # 生成可读的文本报告
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("DINOv3 Classification Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy: {evaluation_results['accuracy']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(evaluation_results['confusion_matrix']) + "\n\n")
        f.write("Classification Report:\n")
        f.write(evaluation_results['classification_report'] + "\n\n")
        f.write("="*60 + "\n")
        f.write("Individual Image Results:\n")
        f.write("="*60 + "\n\n")
        
        for info in test_info:
            status = "✓ CORRECT" if info['correct'] else "✗ WRONG"
            f.write(f"{info['filename']}\n")
            f.write(f"  True Label: {info['label_name']}\n")
            f.write(f"  Predicted: {info['predicted_name']}\n")
            f.write(f"  Status: {status}\n\n")
    
    print(f"Report saved to: {report_path}")


# ===========================
# 8. cls_token可视化
# ===========================
def visualize_cls_token_tsne(features, labels, predictions, output_dir, filename='cls_token_tsne.png'):
    """使用t-SNE可视化cls_token在2D空间的分布"""
    print("\nGenerating t-SNE visualization of cls_token...")
    
    # 确保predictions是numpy数组
    predictions = np.array(predictions)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1), max_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 左图: 按真实标签着色
    mask_1T = labels == 0
    mask_2H = labels == 1
    
    ax1.scatter(features_2d[mask_1T, 0], features_2d[mask_1T, 1], 
                c='red', label='1T (True)', alpha=0.7, s=100, edgecolors='black', linewidths=1.5)
    ax1.scatter(features_2d[mask_2H, 0], features_2d[mask_2H, 1], 
                c='blue', label='2H (True)', alpha=0.7, s=100, edgecolors='black', linewidths=1.5)
    
    ax1.set_title('t-SNE: True Labels\n(384D cls_token → 2D)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 右图: 按预测标签着色（修复版本）
    correct = labels == predictions
    incorrect = ~correct
    
    pred_1T = predictions == 0
    pred_2H = predictions == 1
    
    # 分别绘制预测正确的1T和2H
    correct_1T = correct & pred_1T
    correct_2H = correct & pred_2H
    
    if correct_1T.any():
        ax2.scatter(features_2d[correct_1T, 0], features_2d[correct_1T, 1],
                    c='red', label='Predicted 1T (Correct)', alpha=0.7, s=100, edgecolors='black', linewidths=1.5)
    
    if correct_2H.any():
        ax2.scatter(features_2d[correct_2H, 0], features_2d[correct_2H, 1],
                    c='blue', label='Predicted 2H (Correct)', alpha=0.7, s=100, edgecolors='black', linewidths=1.5)
    
    # 绘制预测错误的样本
    if incorrect.any():
        ax2.scatter(features_2d[incorrect, 0], features_2d[incorrect, 1],
                    c='yellow', marker='x', s=200, label='Misclassified', edgecolors='black', linewidths=2)
    
    ax2.set_title('t-SNE: Predicted Labels\n(384D cls_token → 2D)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE visualization saved to: {save_path}")


def visualize_cls_token_pca(features, labels, predictions, output_dir, filename='cls_token_pca.png'):
    """使用PCA可视化cls_token在2D空间的分布"""
    print("\nGenerating PCA visualization of cls_token...")
    
    # 确保predictions是numpy数组
    predictions = np.array(predictions)
    
    # PCA降维
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    explained_var = pca.explained_variance_ratio_
    print(f"  PC1 explains {explained_var[0]:.2%} variance")
    print(f"  PC2 explains {explained_var[1]:.2%} variance")
    print(f"  Total: {explained_var.sum():.2%} variance")
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 左图: 按真实标签着色
    mask_1T = labels == 0
    mask_2H = labels == 1
    
    ax1.scatter(features_2d[mask_1T, 0], features_2d[mask_1T, 1], 
                c='red', label='1T (True)', alpha=0.7, s=100, edgecolors='black', linewidths=1.5)
    ax1.scatter(features_2d[mask_2H, 0], features_2d[mask_2H, 1], 
                c='blue', label='2H (True)', alpha=0.7, s=100, edgecolors='black', linewidths=1.5)
    
    ax1.set_title(f'PCA: True Labels\n(384D cls_token → 2D, {explained_var.sum():.1%} variance)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=12)
    ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 右图: 按预测标签着色（修复版本）
    correct = labels == predictions
    incorrect = ~correct
    
    pred_1T = predictions == 0
    pred_2H = predictions == 1
    
    # 分别绘制预测正确的1T和2H
    correct_1T = correct & pred_1T
    correct_2H = correct & pred_2H
    
    if correct_1T.any():
        ax2.scatter(features_2d[correct_1T, 0], features_2d[correct_1T, 1],
                    c='red', label='Predicted 1T (Correct)', alpha=0.7, s=100, edgecolors='black', linewidths=1.5)
    
    if correct_2H.any():
        ax2.scatter(features_2d[correct_2H, 0], features_2d[correct_2H, 1],
                    c='blue', label='Predicted 2H (Correct)', alpha=0.7, s=100, edgecolors='black', linewidths=1.5)
    
    # 绘制预测错误的样本
    if incorrect.any():
        ax2.scatter(features_2d[incorrect, 0], features_2d[incorrect, 1],
                    c='yellow', marker='x', s=200, label='Misclassified', edgecolors='black', linewidths=2)
    
    ax2.set_title(f'PCA: Predicted Labels\n(384D cls_token → 2D, {explained_var.sum():.1%} variance)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=12)
    ax2.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"PCA visualization saved to: {save_path}")


def analyze_feature_separation(features, labels, output_dir):
    """分析特征空间的分离度"""
    print("\n" + "="*60)
    print("cls_token Feature Space Analysis")
    print("="*60)
    
    features_1T = features[labels == 0]
    features_2H = features[labels == 1]
    
    # 计算类内距离
    def intra_class_distance(feat):
        n = len(feat)
        if n < 2:
            return 0
        distances = []
        for i in range(min(n, 100)):  # 限制样本数以加快计算
            for j in range(i+1, min(n, 100)):
                dist = np.linalg.norm(feat[i] - feat[j])
                distances.append(dist)
        return np.mean(distances) if distances else 0
    
    # 计算类间距离
    def inter_class_distance(feat1, feat2):
        distances = []
        n1, n2 = len(feat1), len(feat2)
        for i in range(min(n1, 50)):
            for j in range(min(n2, 50)):
                dist = np.linalg.norm(feat1[i] - feat2[j])
                distances.append(dist)
        return np.mean(distances) if distances else 0
    
    intra_1T = intra_class_distance(features_1T)
    intra_2H = intra_class_distance(features_2H)
    inter_dist = inter_class_distance(features_1T, features_2H)
    
    avg_intra = (intra_1T + intra_2H) / 2
    separation_ratio = inter_dist / avg_intra if avg_intra > 0 else 0
    
    print(f"1T class intra-distance:     {intra_1T:.4f}")
    print(f"2H class intra-distance:     {intra_2H:.4f}")
    print(f"Inter-class distance:        {inter_dist:.4f}")
    print(f"Separation ratio:            {separation_ratio:.4f}")
    print()
    if separation_ratio > 2:
        print("✅ Excellent separation (ratio > 2)")
    elif separation_ratio > 1:
        print("✅ Good separation (ratio > 1)")
    else:
        print("⚠️  Poor separation (ratio < 1)")
    print("="*60)
    
    # 保存分析结果
    analysis_path = os.path.join(output_dir, 'feature_space_analysis.txt')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("cls_token Feature Space Analysis\n")
        f.write("="*60 + "\n\n")
        f.write(f"1T class intra-distance:     {intra_1T:.4f}\n")
        f.write(f"2H class intra-distance:     {intra_2H:.4f}\n")
        f.write(f"Inter-class distance:        {inter_dist:.4f}\n")
        f.write(f"Separation ratio:            {separation_ratio:.4f}\n\n")
        f.write("Interpretation:\n")
        f.write("- Separation ratio > 2: Excellent separation\n")
        f.write("- Separation ratio > 1: Good separation (inter-class > intra-class)\n")
        f.write("- Separation ratio < 1: Poor separation\n")
    
    print(f"Analysis saved to: {analysis_path}")


# ===========================
# 9. 主程序
# ===========================
def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载DINOv3模型
    print("Loading DINOv3 model...")
    model = get_dino_model(args.repo_dir, args.model_name, args.weights_path)
    feature_extractor = FeatureExtractor(model, device)
    print("Model loaded!\n")
    
    # 2. 加载训练数据并提取特征
    print("Extracting training features...")
    X_train_1T, y_train_1T, _ = load_and_extract_features(
        args.train_1T_dir, label=0, feature_extractor=feature_extractor
    )
    X_train_2H, y_train_2H, _ = load_and_extract_features(
        args.train_2H_dir, label=1, feature_extractor=feature_extractor
    )
    
    X_train = np.vstack([X_train_1T, X_train_2H])
    y_train = np.concatenate([y_train_1T, y_train_2H])
    print(f"Training set: {len(X_train)} samples\n")
    
    # 3. 训练MLP分类器
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    mlp_model = MLPClassifierModel(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        num_classes=2,
        dropout=args.dropout,
        lr=args.lr,
        device=device
    )
    mlp_model.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
    
    # 4. 先提取测试特征（不生成可视化）
    print("\nExtracting test features...")
    X_test_1T, y_test_1T, _ = load_and_extract_features(
        args.test_1T_dir, label=0, 
        feature_extractor=feature_extractor
    )
    X_test_2H, y_test_2H, _ = load_and_extract_features(
        args.test_2H_dir, label=1,
        feature_extractor=feature_extractor
    )
    
    X_test = np.vstack([X_test_1T, X_test_2H])
    y_test = np.concatenate([y_test_1T, y_test_2H])
    print(f"Test set: {len(X_test)} samples\n")
    
    # 5. 评估模型并获取预测
    evaluation_results = mlp_model.evaluate(X_test, y_test)
    predictions = evaluation_results['predictions']
    
    # 6. 生成带预测结果的PCA可视化
    print("\nGenerating PCA visualizations with predictions...")
    pca_visualizer = PCAVisualizer(args.output_dir)
    
    # 分别处理1T和2H，传入对应的预测结果
    n_1T = len(X_test_1T)
    predictions_1T = predictions[:n_1T]
    predictions_2H = predictions[n_1T:]
    
    _, _, info_1T = load_and_extract_features(
        args.test_1T_dir, label=0,
        feature_extractor=feature_extractor,
        pca_visualizer=pca_visualizer,
        predictions=predictions_1T,
        n_train_samples=len(X_train)
    )
    _, _, info_2H = load_and_extract_features(
        args.test_2H_dir, label=1,
        feature_extractor=feature_extractor,
        pca_visualizer=pca_visualizer,
        predictions=predictions_2H,
        n_train_samples=len(X_train)
    )
    
    test_info = info_1T + info_2H
    
    # 7. 保存结果
    print("\nSaving results...")
    save_results(args.output_dir, test_info, predictions, evaluation_results)
    
    # 8. 可视化cls_token特征空间分布
    print("\n" + "="*60)
    print("Visualizing cls_token Feature Space")
    print("="*60)
    
    # 分析特征分离度
    analyze_feature_separation(X_test, y_test, args.output_dir)
    
    # t-SNE可视化
    visualize_cls_token_tsne(X_test, y_test, predictions, args.output_dir)
    
    # PCA可视化
    visualize_cls_token_pca(X_test, y_test, predictions, args.output_dir)
    
    print("\n" + "="*60)
    print("All done! Check the output directory for results:")
    print(f"  - Combined images (Original + PCA) with predictions (*_combined.png)")
    print(f"  - cls_token t-SNE visualization (cls_token_tsne.png)")
    print(f"  - cls_token PCA visualization (cls_token_pca.png)")
    print(f"  - Feature space analysis (feature_space_analysis.txt)")
    print(f"  - classification_results.json")
    print(f"  - classification_report.txt")
    print(f"\nTraining Info:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Model: MLP Neural Network")
    print(f"  - Hidden layers: {args.hidden_dims}")
    print(f"  - Training epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print("="*60)


if __name__ == "__main__":
    main()
