# -*- coding: utf-8 -*-
"""
EEG-Conformer + 序列建模 (Stage 2)
结合 EEG-Conformer 的特征提取 + Bi-LSTM 序列建模
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from conformer import Conformer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ConformerSeq(nn.Module):
    """
    EEG-Conformer + Bi-LSTM 序列建模
    
    Stage 1: 使用预训练的 Conformer 提取特征
    Stage 2: Bi-LSTM 进行序列建模
    """
    
    def __init__(self, n_classes=5, seq_length=25, hidden_size=256, 
                 n_rnn_layers=2, dropout=0.5, pretrained_conformer=None):
        super().__init__()
        
        self.seq_length = seq_length
        self.n_classes = n_classes
        
        # Conformer 特征提取器
        self.conformer = Conformer(emb_size=40, depth=6, n_classes=n_classes, n_electrodes=1)
        
        if pretrained_conformer is not None:
            self.conformer.load_state_dict(pretrained_conformer.state_dict())
        
        # 获取 Conformer 的特征维度 (需要前向传播一次)
        self.feature_dim = None  # 将在第一次前向传播时确定
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=256,  # Conformer fc 输出是 256
            hidden_size=hidden_size,
            num_layers=n_rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_rnn_layers > 1 else 0
        )
        
        # 输出层（残差连接）
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size * 2 + 256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
    
    def extract_features(self, x):
        """提取 Conformer 中间特征"""
        # x: (batch, 1, 1, 3000)
        # 通过 Conformer 的前几层获取 token embedding
        x = self.conformer[0](x)  # PatchEmbedding
        x = self.conformer[1](x)  # TransformerEncoder
        # x: (batch, n_tokens, emb_size)
        
        # 通过 ClassificationHead 的第一部分获取特征
        cls_head = self.conformer[2]
        x = x.contiguous().view(x.size(0), -1)  # Flatten
        
        # 只用前面的层提取特征，不做最后的分类
        x = cls_head.fc[0](x)  # LazyLinear -> 256
        x = cls_head.fc[1](x)  # ELU
        x = cls_head.fc[2](x)  # Dropout
        
        return x  # (batch, 256)
    
    def forward(self, x, return_all=True):
        """
        Args:
            x: (batch, seq_length, 1, 1, 3000) 或 (batch, seq_length, 3000)
        Returns:
            logits: (batch, seq_length, n_classes)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 确保输入格式正确
        if x.dim() == 3:
            # (batch, seq, 3000) -> (batch, seq, 1, 1, 3000)
            x = x.unsqueeze(2).unsqueeze(3)
        
        # 展平 batch 和 sequence
        x = x.view(-1, 1, 1, x.size(-1))  # (batch*seq, 1, 1, 3000)
        
        # 提取 Conformer 特征
        features = self.extract_features(x)  # (batch*seq, 256)
        
        # 重塑为序列
        features_seq = features.view(batch_size, seq_len, -1)  # (batch, seq, 256)
        
        # Bi-LSTM
        lstm_out, _ = self.lstm(features_seq)  # (batch, seq, hidden*2)
        
        # 残差连接 + 分类
        if return_all:
            # 所有时间步
            lstm_out_flat = lstm_out.contiguous().view(-1, lstm_out.size(-1))
            combined = torch.cat([lstm_out_flat, features], dim=1)
            logits = self.output_fc(combined)
            logits = logits.view(batch_size, seq_len, -1)
        else:
            # 只返回最后时间步
            last_lstm = lstm_out[:, -1, :]
            last_feat = features.view(batch_size, seq_len, -1)[:, -1, :]
            combined = torch.cat([last_lstm, last_feat], dim=1)
            logits = self.output_fc(combined)
        
        return logits


class SleepSeqDataset(Dataset):
    """序列数据集"""
    
    def __init__(self, npz_files, seq_length=25):
        import glob
        self.seq_length = seq_length
        self.sequences = []
        self.labels = []
        
        for f in npz_files:
            data = np.load(f)
            x = data['x']
            y = data['y']
            
            if x.ndim == 3:
                x = x.squeeze(-1)
            
            n_epochs = len(y)
            for i in range(n_epochs - seq_length + 1):
                self.sequences.append(x[i:i+seq_length].astype(np.float32))
                self.labels.append(y[i:i+seq_length].astype(np.int64))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx]  # (seq, 3000)
        y = self.labels[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def get_fold_files(data_dir, fold_idx):
    import glob
    import re
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    val_files, train_files = [], []
    for f in all_files:
        match = re.match(r'SC4(\d{2})\d+E0\.npz', os.path.basename(f))
        if match:
            if int(match.group(1)) == fold_idx:
                val_files.append(f)
            else:
                train_files.append(f)
    return sorted(train_files), sorted(val_files)


def train_conformer_seq(data_dir, fold_idx, output_dir, pretrained_path=None,
                        n_epochs=100, seq_length=25, batch_size=16, lr=1e-4, seed=42):
    """训练 Conformer + 序列建模"""
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    fold_dir = os.path.join(output_dir, f'fold_{fold_idx:02d}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # 数据
    train_files, val_files = get_fold_files(data_dir, fold_idx)
    print(f"Fold {fold_idx}: {len(train_files)} train files, {len(val_files)} val files")
    
    train_dataset = SleepSeqDataset(train_files, seq_length)
    val_dataset = SleepSeqDataset(val_files, seq_length)
    print(f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 加载预训练的 Conformer
    pretrained_conformer = None
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained Conformer from {pretrained_path}")
        pretrained_conformer = Conformer(emb_size=40, depth=6, n_classes=5, n_electrodes=1)
        pretrained_conformer.eval()
        with torch.no_grad():
            dummy = torch.randn(2, 1, 1, 3000)
            _ = pretrained_conformer(dummy)
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        pretrained_conformer.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建模型
    model = ConformerSeq(
        n_classes=5, seq_length=seq_length, hidden_size=256,
        n_rnn_layers=2, dropout=0.5, pretrained_conformer=pretrained_conformer
    ).to(device)
    
    # 初始化
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(2, seq_length, 3000).to(device)
        _ = model(dummy)
    model.train()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 不同学习率
    conformer_params = list(model.conformer.parameters())
    lstm_params = list(model.lstm.parameters()) + list(model.output_fc.parameters())
    optimizer = optim.Adam([
        {'params': conformer_params, 'lr': lr * 0.1},  # Conformer 用更小的学习率
        {'params': lstm_params, 'lr': lr}
    ])
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_results = None
    
    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits.view(-1, 5), y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.numel()
            train_correct += (logits.argmax(-1) == y).sum().item()
            train_total += y.numel()
        
        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                all_preds.extend(logits.argmax(-1).view(-1).cpu().numpy())
                all_labels.extend(y.view(-1).cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_acc = (all_preds == all_labels).mean()
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_kappa = cohen_kappa_score(all_labels, all_preds)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{n_epochs} | "
                  f"Train Loss: {train_loss/train_total:.4f}, Acc: {train_correct/train_total:.4f} | "
                  f"Val Acc: {val_acc:.4f}, MF1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_results = {
                'acc': val_acc, 'macro_f1': val_f1, 'kappa': val_kappa,
                'y_true': all_labels, 'y_pred': all_preds
            }
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, os.path.join(fold_dir, 'best_model.pth'))
    
    print(f"\nBest Val Acc: {best_results['acc']:.4f}, MF1: {best_results['macro_f1']:.4f}, Kappa: {best_results['kappa']:.4f}")
    
    np.savez(os.path.join(fold_dir, 'final_results.npz'), **best_results)
    return best_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/Sleep-EDF-20/eeg_fpz_cz")
    parser.add_argument("--fold-idx", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/conformer_seq")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained Conformer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seq-length", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    train_conformer_seq(
        data_dir=args.data_dir,
        fold_idx=args.fold_idx,
        output_dir=args.output_dir,
        pretrained_path=args.pretrained,
        n_epochs=args.epochs,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

