# -*- coding: utf-8 -*-
"""
Sleep-EDF-20 数据加载器
支持 20-fold 受试者级别交叉验证（与 DeepSleepNet 完全一致）
"""

import glob
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


SLEEPEDF_RECORD_PATTERN = re.compile(r"SC4(?P<subject>\d{2})\d+E0(?:\.npz)?$")


def parse_sleepedf_record(path):
    """Return the stable subject and record identifiers encoded by a file name.

    The record identifier is the NPZ stem (for example ``SC4001E0``).  Epoch
    identity is completed by the zero-based index within that record.  Parsing
    is deliberately strict so a new naming convention cannot silently change
    subject membership.
    """
    record_id = Path(path).stem
    match = SLEEPEDF_RECORD_PATTERN.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Unrecognized Sleep-EDF record name: {Path(path).name}")
    return int(match.group("subject")), record_id


class SleepEDFDataset(Dataset):
    """
    Sleep-EDF-20 PyTorch Dataset

    加载 .npz 文件并转换为 EEG-Conformer 所需的格式
    输出格式: (1, 1, 3000) - (conv_channel, electrodes, time_samples)
    """

    def __init__(
        self,
        npz_files,
        transform=None,
        *,
        return_identity=False,
        dataset_id="sleepedf20",
        outer_fold=None,
    ):
        """
        Args:
            npz_files: list of .npz file paths
            transform: optional data augmentation
            return_identity: when true, return a stable row metadata mapping
                in addition to ``(x, y)``.
            dataset_id: immutable dataset name stored with every row.
            outer_fold: outer-fold identifier stored with every row.  It is
                required when ``return_identity`` is true.
        """
        self.data_list = []
        self.label_list = []
        self.file_indices = []  # 记录每个样本来自哪个文件
        self.identities = []
        self.return_identity = bool(return_identity)
        self.dataset_id = str(dataset_id)
        self.outer_fold = None if outer_fold is None else int(outer_fold)

        if self.return_identity and self.outer_fold is None:
            raise ValueError("outer_fold is required when return_identity=True")

        for file_idx, f in enumerate(npz_files):
            data = np.load(f)
            x = data["x"]  # (n_epochs, 3000) 或 (n_epochs, 3000, 1)
            y = data["y"]  # (n_epochs,)

            # 确保 x 是 2D: (n_epochs, 3000)
            if x.ndim == 3:
                x = x.squeeze(-1)

            # 转换为 Conformer 输入格式: (n_epochs, 1, 1, 3000)
            # 维度: (batch, conv_channel, electrodes, time_samples)
            x = x[:, np.newaxis, np.newaxis, :]

            self.data_list.append(x)
            self.label_list.append(y)
            self.file_indices.extend([file_idx] * len(y))

            if self.return_identity:
                subject_id, record_id = parse_sleepedf_record(f)
                self.identities.extend(
                    {
                        "dataset": self.dataset_id,
                        "subject_id": subject_id,
                        "record_id": record_id,
                        "epoch_index": int(epoch_index),
                        "outer_fold": self.outer_fold,
                        "ground_truth_label": int(label),
                    }
                    for epoch_index, label in enumerate(y)
                )

        self.data = np.concatenate(self.data_list, axis=0).astype(np.float32)
        self.labels = np.concatenate(self.label_list, axis=0).astype(np.int64)
        self.transform = transform

        # 数据标准化（per-sample z-score）
        # 注意：DeepSleepNet 的预处理脚本中数据已经标准化，这里可以选择跳过
        # 如果需要额外标准化，取消下面的注释
        # self._normalize()

    def _normalize(self):
        """对每个样本进行 z-score 标准化"""
        for i in range(len(self.data)):
            sample = self.data[i]
            mean = sample.mean()
            std = sample.std()
            if std > 0:
                self.data[i] = (sample - mean) / std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        sample = torch.from_numpy(x.copy())
        label = torch.tensor(y, dtype=torch.long)
        if not self.return_identity:
            return sample, label
        return sample, label, dict(self.identities[idx])

    def get_identities(self):
        """Return stable metadata in dataset order without exposing internals."""
        if not self.return_identity:
            raise RuntimeError("Dataset was created without return_identity=True")
        return [dict(identity) for identity in self.identities]

    def get_class_distribution(self):
        """返回类别分布"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


def get_fold_files(data_dir, fold_idx):
    """
    获取指定 fold 的训练集和验证集文件

    与 DeepSleepNet 完全一致的分割逻辑：
    - 基于文件名模式匹配
    - 验证集：当前 fold 对应的受试者（fold_idx 即受试者 ID）
    - 训练集：其他所有受试者

    文件命名规则: SC4{subject_id:02d}{night}E0.npz
    例如: SC4001E0.npz → 受试者 00, 夜晚 1
          SC4012E0.npz → 受试者 01, 夜晚 2
          SC4131E0.npz → 受试者 13, 夜晚 1（只有一晚）

    Args:
        data_dir: 数据目录路径
        fold_idx: fold 索引 (0-19)

    Returns:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    val_files = []
    train_files = []

    for f in all_files:
        basename = os.path.basename(f)
        # 提取受试者 ID：SC4{XX}{Y}E0.npz 中的 XX
        match = SLEEPEDF_RECORD_PATTERN.fullmatch(basename)
        if match:
            subject_id = int(match.group("subject"))
            if subject_id == fold_idx:
                val_files.append(f)
            else:
                train_files.append(f)

    return sorted(train_files), sorted(val_files)


def create_dataloaders(data_dir, fold_idx, batch_size=64, num_workers=0, verbose=True):
    """
    创建训练和验证 DataLoader

    Args:
        data_dir: 数据目录路径
        fold_idx: fold 索引 (0-19)
        batch_size: 批次大小
        num_workers: DataLoader 工作进程数（Windows 建议设为 0）
        verbose: 是否打印详细信息

    Returns:
        train_loader: 训练集 DataLoader
        val_loader: 验证集 DataLoader
    """
    train_files, val_files = get_fold_files(data_dir, fold_idx)

    if verbose:
        print(f"\n========== [Fold-{fold_idx:02d}] ==========")
        print(f"Training files ({len(train_files)}):")
        for f in train_files:
            print(f"  - {os.path.basename(f)}")
        print(f"\nValidation files ({len(val_files)}):")
        for f in val_files:
            print(f"  - {os.path.basename(f)}")

    train_dataset = SleepEDFDataset(train_files)
    val_dataset = SleepEDFDataset(val_files)

    if verbose:
        print(f"\nTrain samples: {len(train_dataset)}")
        print(f"Train class distribution: {train_dataset.get_class_distribution()}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Val class distribution: {val_dataset.get_class_distribution()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader


def compute_class_weights(data_dir, fold_idx):
    """
    计算类别权重用于 class-weighted loss

    权重公式: w_i = N / (C * n_i)
    其中 N = 总样本数, C = 类别数, n_i = 第 i 类样本数

    这种方法使少数类获得更高权重，帮助模型关注难以学习的类别（如 N1）

    Args:
        data_dir: 数据目录路径
        fold_idx: fold 索引

    Returns:
        weights: torch.FloatTensor of shape (n_classes,)
    """
    train_files, _ = get_fold_files(data_dir, fold_idx)

    all_labels = []
    for f in train_files:
        data = np.load(f)
        y = data["y"]
        all_labels.append(y)

    labels = np.concatenate(all_labels)

    n_classes = 5  # W, N1, N2, N3, REM
    class_counts = np.bincount(labels, minlength=n_classes)
    total = len(labels)

    # 计算权重: 样本越少的类别权重越大
    weights = total / (n_classes * class_counts + 1e-6)

    # 归一化使平均权重为 1
    weights = weights / weights.mean()

    return torch.FloatTensor(weights)


if __name__ == "__main__":
    # 测试代码
    import sys

    data_dir = "data/Sleep-EDF-20/eeg_fpz_cz"

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # 测试 fold 0
    print("Testing data loader for Fold 0...")
    train_loader, val_loader = create_dataloaders(data_dir, fold_idx=0, batch_size=64)

    # 获取一个 batch 验证形状
    for X, y in train_loader:
        print(f"\nBatch X shape: {X.shape}")  # 应该是 (batch, 1, 1, 3000)
        print(f"Batch y shape: {y.shape}")
        print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
        print(f"X value range: [{X.min():.4f}, {X.max():.4f}]")
        print(f"Unique labels in batch: {torch.unique(y).tolist()}")
        break

    # 测试类别权重
    weights = compute_class_weights(data_dir, fold_idx=0)
    print(f"\nClass weights: {weights.numpy()}")
