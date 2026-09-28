# -*- coding: utf-8 -*-
"""
单个 Fold 的训练脚本
用于 EEG-Conformer 在 Sleep-EDF-20 上的 20-fold 交叉验证
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conformer import Conformer
from sleepedf.data_loader import create_dataloaders, compute_class_weights


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        _, logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, device, tau_h=0.9):
    """
    评估模型，包含 High-Confidence Error (HCE) 分析
    
    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
        tau_h: HCE 置信度阈值 (default=0.9)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []  # HCE: 收集每个样本的置信度
    all_probs = []  # Paper2: 收集完整的 softmax 概率分布
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            _, logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            
            # HCE: 计算 softmax 置信度 = max(softmax probability)
            probs = torch.softmax(logits, dim=1)
            confidences, preds = probs.max(dim=1)  # 同时获取置信度和预测
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_probs.append(probs.cpu().numpy())  # Paper2: 保存完整概率
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_probs = np.vstack(all_probs)  # Paper2: (N, 5) 概率分布
    
    # 原有指标
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    avg_loss = total_loss / len(all_labels)
    
    # HCE: 识别高置信度错误
    # HCE = (预测错误) AND (置信度 >= tau_h)
    is_error = (all_preds != all_labels)
    is_high_confidence = (all_confidences >= tau_h)
    hce_mask = is_error & is_high_confidence
    
    # HCE 统计
    total_samples = len(all_labels)
    hce_count = hce_mask.sum()
    hce_rate = hce_count / total_samples  # HCE 占总样本的比例
    
    return {
        # 原有输出
        'loss': avg_loss,
        'acc': acc,
        'macro_f1': macro_f1,
        'kappa': kappa,
        'confusion_matrix': cm,
        'y_true': all_labels,
        'y_pred': all_preds,
        # HCE 相关输出
        'confidences': all_confidences,      # 所有样本的置信度
        'probs': all_probs,                  # Paper2: 完整 softmax 概率分布 (N, 5)
        'hce_rate': float(hce_rate),         # HCE 比例
        'hce_count': int(hce_count),         # HCE 数量
        'total_samples': total_samples,      # 总样本数
        'hce_mask': hce_mask,                # HCE 样本的 mask
        'tau_h': tau_h                       # 使用的阈值
    }


def train_fold(
    data_dir,
    fold_idx,
    output_dir,
    n_epochs=100,
    batch_size=64,
    lr=1e-3,
    early_stop_patience=20,
    use_class_weight=True,
    seed=42,
    verbose=True
):
    """
    训练单个 fold
    
    Args:
        data_dir: 数据目录路径
        fold_idx: fold 索引 (0-19)
        output_dir: 输出目录
        n_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        early_stop_patience: 早停耐心值，0 表示禁用
        use_class_weight: 是否使用类别权重
        seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        final_results: 最终评估结果字典
    """
    
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx:02d}")
    os.makedirs(fold_output_dir, exist_ok=True)
    
    # 创建数据加载器（Windows 上 num_workers=0 更稳定）
    train_loader, val_loader = create_dataloaders(
        data_dir, fold_idx, batch_size=batch_size, num_workers=0, verbose=verbose
    )
    
    # 计算类别权重
    if use_class_weight:
        class_weights = compute_class_weights(data_dir, fold_idx).to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 创建模型
    # Sleep-EDF: 单通道, 3000 采样点
    model = Conformer(
        emb_size=40,
        depth=6,
        n_classes=5,
        n_electrodes=1  # 单通道 EEG
    ).to(device)
    
    # 初始化 LazyLinear（需要先跑一个 forward pass）
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 1, 3000).to(device)
        _ = model(dummy_input)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # 训练循环
    best_val_acc = 0.0
    best_val_f1 = 0.0
    epochs_no_improve = 0
    history = []
    
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        val_results = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        print(
            f"Epoch {epoch:03d}/{n_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_results['loss']:.4f}, Acc: {val_results['acc']:.4f}, "
            f"MF1: {val_results['macro_f1']:.4f}, Kappa: {val_results['kappa']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.1f}s"
        )
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_results['loss'],
            'val_acc': val_results['acc'],
            'val_macro_f1': val_results['macro_f1'],
            'val_kappa': val_results['kappa']
        })
        
        # 保存最佳模型（基于验证集准确率）
        if val_results['acc'] > best_val_acc:
            best_val_acc = val_results['acc']
            best_val_f1 = val_results['macro_f1']
            epochs_no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_f1': best_val_f1,
                'confusion_matrix': val_results['confusion_matrix']
            }, os.path.join(fold_output_dir, 'best_model.pth'))
            
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break
    
    # 保存训练历史
    np.save(os.path.join(fold_output_dir, 'history.npy'), history)
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load(os.path.join(fold_output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_results = evaluate(model, val_loader, device, tau_h=0.9)
    
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx:02d} Final Results (Best Model)")
    print(f"{'='*50}")
    print(f"Accuracy:  {final_results['acc']:.4f}")
    print(f"Macro-F1:  {final_results['macro_f1']:.4f}")
    print(f"Kappa:     {final_results['kappa']:.4f}")
    # HCE 统计
    print(f"\n--- High-Confidence Error (HCE) Analysis ---")
    print(f"Confidence threshold (tau_h): {final_results['tau_h']}")
    print(f"HCE Count:    {final_results['hce_count']} / {final_results['total_samples']}")
    print(f"HCE Rate:     {final_results['hce_rate']:.4f} ({final_results['hce_rate']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"       W    N1    N2    N3   REM")
    for i, row in enumerate(final_results['confusion_matrix']):
        stage = ['W', 'N1', 'N2', 'N3', 'REM'][i]
        print(f"{stage:>3} {row}")
    
    # 保存最终结果
    np.savez(
        os.path.join(fold_output_dir, 'final_results.npz'),
        acc=final_results['acc'],
        macro_f1=final_results['macro_f1'],
        kappa=final_results['kappa'],
        confusion_matrix=final_results['confusion_matrix'],
        y_true=final_results['y_true'],
        y_pred=final_results['y_pred'],
        # HCE 相关
        confidences=final_results['confidences'],
        probs_T=final_results['probs'],       # Paper2: 完整 softmax 概率分布
        hce_rate=final_results['hce_rate'],
        hce_count=final_results['hce_count'],
        total_samples=final_results['total_samples'],
        tau_h=final_results['tau_h']
    )
    
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Train EEG-Conformer on Sleep-EDF-20 (single fold)"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data/Sleep-EDF-20/eeg_fpz_cz",
        help="Path to preprocessed .npz files"
    )
    parser.add_argument(
        "--fold-idx", 
        type=int, 
        required=True,
        help="Fold index (0-19)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs/sleepedf20",
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--early-stop", 
        type=int, 
        default=20,
        help="Early stopping patience (0 to disable)"
    )
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable class-weighted loss"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    train_fold(
        data_dir=args.data_dir,
        fold_idx=args.fold_idx,
        output_dir=args.output_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stop_patience=args.early_stop,
        use_class_weight=not args.no_class_weight,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

