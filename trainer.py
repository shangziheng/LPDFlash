"""
训练器 - 彩色偏振双分支去噪网络
输入: 12通道 (4方向 × RGB)
输出:
  - 分支1: S0彩色图像 [B, 3, H, W]
  - 分支2: 灰度偏振图像 [B, 4, H, W]
"""
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import kornia
from models.LPDFlash import LPDFlash
import os
import math
import random
import argparse
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class BranchDataset(Dataset):
    """彩色偏振去噪数据集 - 输出12通道 (4方向 × RGB)"""

    def __init__(self, clean_dir, noisy_dir, mode='train', image_size=1024, crop_size=256, stride=128,
                 normalize_brightness=False, target_brightness=0.1, use_pt=True):
        """
        Args:
            clean_dir: 清晰图像目录（.pt 文件目录或原始图片目录）
            noisy_dir: 噪声图像目录
            mode: 'train', 'val', 'test'
            image_size: 输入图像大小
            crop_size: 裁剪大小
            stride: 裁剪步长
            normalize_brightness: 是否对输入进行亮度归一化
            target_brightness: 归一化的目标平均亮度
            use_pt: 是否使用 .pt 预处理文件（推荐True）
        """
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.mode = mode
        self.image_size = image_size
        self.crop_size = crop_size
        self.stride = stride
        self.eps = 1e-8
        self.normalize_brightness = normalize_brightness
        self.target_brightness = target_brightness
        self.use_pt = use_pt

        # RGB转灰度权重 (R, G, B顺序，数据是RGB格式)
        self.rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

        self.do_crop = (mode in ['train', 'val', 'test'])  # test模式也裁剪
        self.center_crop_size = 1024

        # 检测数据格式
        if self.use_pt:
            self.sample_files = self._find_pt_files()
            print(f"[SingleBranch] 使用 .pt 文件，找到 {len(self.sample_files)} 个样本")
        else:
            self.sample_folders = self._find_valid_sample_folders()
            print(f"[SingleBranch] 使用原始图片，找到 {len(self.sample_folders)} 个样本")

        if self.normalize_brightness:
            print(f"[SingleBranch] 启用输入亮度归一化（仅S0彩色通道），目标亮度 = {self.target_brightness}")

        # 计算裁剪相关参数
        if self.do_crop:
            self.num_crops_per_image = self._calculate_num_crops()
            self.crop_positions = self._precompute_crop_positions()
        else:
            self.num_crops_per_image = 1
            self.crop_positions = None

        print(f"初始化完成，数据集大小: {len(self)}")
        print(f"[INFO] 输出格式: 12通道 (4方向 × RGB)")

    def _find_pt_files(self):
        """查找所有 .pt 文件"""
        if not os.path.exists(self.clean_dir):
            raise FileNotFoundError(f"clean_dir 不存在: {self.clean_dir}")
        pt_files = sorted([f for f in os.listdir(self.clean_dir) if f.endswith('.pt')])
        return [os.path.join(self.clean_dir, f) for f in pt_files]

    def _find_valid_sample_folders(self):
        """找到所有有效的样本文件夹（兼容模式）"""
        sample_folders = []
        if not os.path.exists(self.clean_dir):
            raise FileNotFoundError(f"clean_dir 不存在: {self.clean_dir}")
        if not os.path.exists(self.noisy_dir):
            raise FileNotFoundError(f"noisy_dir 不存在: {self.noisy_dir}")

        folders = sorted(os.listdir(self.clean_dir))
        folders = [f for f in folders if os.path.isdir(os.path.join(self.clean_dir, f))]

        for folder in folders:
            clean_folder = os.path.join(self.clean_dir, folder)
            noisy_folder = os.path.join(self.noisy_dir, folder)

            clean_files = [
                os.path.join(clean_folder, "0.png"),
                os.path.join(clean_folder, "45.png"),
                os.path.join(clean_folder, "90.png"),
                os.path.join(clean_folder, "135.png")
            ]
            noisy_files = [
                os.path.join(noisy_folder, "0.png"),
                os.path.join(noisy_folder, "45.png"),
                os.path.join(noisy_folder, "90.png"),
                os.path.join(noisy_folder, "135.png")
            ]

            if all(os.path.exists(f) for f in clean_files + noisy_files):
                sample_folders.append({
                    'folder': folder,
                    'clean_files': clean_files,
                    'noisy_files': noisy_files
                })
        return sample_folders


    def _load_single_image(self, file_path):
        """加载并预处理单张彩色图像（兼容模式）"""
        img_bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"无法读取图像: {file_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = img_rgb.shape

        # 中心裁剪
        start_y = (h - self.center_crop_size) // 2
        start_x = (w - self.center_crop_size) // 2
        img_rgb = img_rgb[start_y:start_y + self.center_crop_size,
                          start_x:start_x + self.center_crop_size, :]

        # 调整大小
        if img_rgb.shape[0] != self.image_size:
            img_rgb = cv2.resize(img_rgb, (self.image_size, self.image_size),
                                 interpolation=cv2.INTER_CUBIC)

        tensor = np.transpose(img_rgb.astype(np.float32) / 255.0, (2, 0, 1))
        return tensor

    def _load_sample_from_pt(self, idx):
        """从 .pt 文件加载样本（必须为12通道格式）"""
        data = torch.load(self.sample_files[idx], weights_only=True)

        data_format = data.get('format', '12ch')  # 默认为12通道
        if data_format == '7ch':
            # 无法从7通道恢复为12通道，抛出错误提示
            raise ValueError(
                f"数据格式为 '7ch'，但数据集要求输出12通道。请使用12通道格式的 .pt 文件，"
                f"或重新生成数据（设置 format='12ch'）。"
            )
        # 直接返回12通道数据
        return data['clean'], data['noisy']

    def _load_sample_from_images(self, idx):
        """从原始图片加载样本，输出12通道（4张RGB拼接）"""
        sample = self.sample_folders[idx]
        clean_tensors = [self._load_single_image(f) for f in sample['clean_files']]
        noisy_tensors = [self._load_single_image(f) for f in sample['noisy_files']]
        clean_12ch = np.concatenate(clean_tensors, axis=0)  # [12, H, W]
        noisy_12ch = np.concatenate(noisy_tensors, axis=0)  # [12, H, W]
        # 直接返回12通道，不再转换为7通道
        return torch.from_numpy(clean_12ch), torch.from_numpy(noisy_12ch)

    def _calculate_num_crops(self):
        """计算每张图像的裁剪块数量"""
        num_x = math.floor((self.image_size - self.crop_size) / self.stride) + 1
        num_y = math.floor((self.image_size - self.crop_size) / self.stride) + 1
        return num_x * num_y

    def _precompute_crop_positions(self):
        """预计算所有裁剪位置"""
        crop_positions = []
        num_x = math.floor((self.image_size - self.crop_size) / self.stride) + 1
        for i in range(num_x):
            for j in range(num_x):
                x = j * self.stride
                y = i * self.stride
                x = min(x, self.image_size - self.crop_size)
                y = min(y, self.image_size - self.crop_size)
                crop_positions.append((x, y))
        return crop_positions

    @staticmethod
    def apply_aop_rotation(tensor, shift):
        if shift == 0:
            return tensor
        shape = tensor.shape
        if tensor.dim() == 3:
            t = tensor.view(4, 3, *shape[1:])
            t = t.roll(shifts=shift, dims=0)
            return t.view(shape)
        elif tensor.dim() == 4:
            t = tensor.view(tensor.size(0), 4, 3, *shape[2:])
            t = t.roll(shifts=shift, dims=1)
            return t.view(shape)
        else:
            raise ValueError("只支持 3D 或 4D 张量")

    def __len__(self):
        if self.use_pt:
            num_samples = len(self.sample_files)
        else:
            num_samples = len(self.sample_folders)

        if self.do_crop:
            return num_samples * self.num_crops_per_image
        else:
            return num_samples

    def __getitem__(self, idx):
        if self.do_crop:
            # 确定样本和裁剪位置
            if self.use_pt:
                num_samples = len(self.sample_files)
            else:
                num_samples = len(self.sample_folders)

            sample_idx = idx // self.num_crops_per_image
            crop_idx = idx % self.num_crops_per_image
            x, y = self.crop_positions[crop_idx]

            # 加载 (12通道)
            if self.use_pt:
                clean, noisy = self._load_sample_from_pt(sample_idx)
            else:
                clean, noisy = self._load_sample_from_images(sample_idx)

            # 裁剪
            clean_crop = clean[:, y:y + self.crop_size, x:x + self.crop_size].clone()
            noisy_crop = noisy[:, y:y + self.crop_size, x:x + self.crop_size].clone()

            # 亮度归一化
            if self.normalize_brightness:
                s0_channels = noisy_crop[:3, :, :]
                current_mean = s0_channels.mean()
                if current_mean > self.eps:
                    scale = self.target_brightness / current_mean
                    noisy_crop[:3, :, :] = torch.clamp(s0_channels * scale, 0.0, 1.0)

            # 训练增强（AOP 旋转）
            if self.mode == 'train':
                shift = random.randint(0, 3)
                clean_crop = self.apply_aop_rotation(clean_crop, shift)
                noisy_crop = self.apply_aop_rotation(noisy_crop, shift)

            return noisy_crop, clean_crop

        else:
            # 无裁剪的测试/全图模式
            if self.use_pt:
                clean, noisy = self._load_sample_from_pt(idx)
            else:
                clean, noisy = self._load_sample_from_images(idx)

            if self.normalize_brightness:
                s0_channels = noisy[:3, :, :]
                current_mean = s0_channels.mean()
                if current_mean > self.eps:
                    scale = self.target_brightness / current_mean
                    noisy[:3, :, :] = torch.clamp(s0_channels * scale, 0.0, 1.0)

            return noisy, clean

class PolarizationLoss(nn.Module):
    """
    损失函数 - 适配 12 通道输入 (4方向×RGB)

    模型输出:
        - branch1_out: [B, 3, H, W]  S0彩色图像
        - branch2_out: [B, 4, H, W]  灰度偏振图像 (I0, I45, I90, I135)

    目标格式:
        - target: [B, 12, H, W] = 4方向 × RGB (I0_RGB, I45_RGB, I90_RGB, I135_RGB)

    损失:
        - L1(S0_pred, S0_target)
        - L1(DoLP_pred, DoLP_target)
        - L1(AoP_pred, AoP_target) 周期性处理
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.eps = 1e-8
        # RGB转灰度权重 (R, G, B顺序，数据是RGB格式)
        self.rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

    def _extract_s0_rgb_and_gray_polar(self, target_12ch):
        """
        从12通道目标中提取 S0彩色(3ch) 和 灰度偏振(4ch)
        Args:
            target_12ch: [B, 12, H, W] (方向顺序: 0°,45°,90°,135°, 每个方向RGB)
        Returns:
            s0_rgb: [B,3,H,W]  四个方向的RGB平均
            gray_polar: [B,4,H,W] 每个方向RGB转灰度
        """
        B, C, H, W = target_12ch.shape
        # 重排为 [B, 4, 3, H, W]
        polar_rgb = target_12ch.view(B, 4, 3, H, W)

        # S0彩色 = 4个方向RGB平均
        s0_rgb = polar_rgb.mean(dim=1)  # [B,3,H,W]

        # 灰度偏振: 每个方向的RGB转灰度
        # polar_rgb: [B,4,3,H,W] * weights [1,3,1,1] -> sum(dim=2) -> [B,4,H,W]
        gray_polar = (polar_rgb * self.rgb_to_gray_weights.to(target_12ch.device)).sum(dim=2)

        return s0_rgb, gray_polar

    def _compute_polarization(self, gray_polar):
        """
        从4通道灰度偏振图像计算 DoLP 和 AoP
        Args:
            gray_polar: [B,4,H,W] (I0, I45, I90, I135)
        Returns:
            DoLP: [B,1,H,W]
            AoP:  [B,1,H,W] 范围 [0, π]
        """
        i0 = gray_polar[:, 0:1, :, :]
        i45 = gray_polar[:, 1:2, :, :]
        i90 = gray_polar[:, 2:3, :, :]
        i135 = gray_polar[:, 3:4, :, :]

        S1 = (i0 - i90) * 0.5
        S2 = (i45 - i135) * 0.5
        S0 = (i0 + i45 + i90 + i135) * 0.25

        DoLP = torch.sqrt(S1 ** 2 + S2 ** 2 + self.eps) / (S0 + self.eps)
        DoLP = torch.clamp(DoLP, 0.0, 1.0)

        AoP = 0.5 * torch.atan2(S2, S1 + self.eps)
        AoP = AoP % math.pi
        return DoLP, AoP

    def forward(self, pred, target_12ch):
        """
        Args:
            pred: tuple (branch1_out, branch2_out)
                - branch1_out: [B,3,H,W]  S0彩色预测
                - branch2_out: [B,4,H,W]  灰度偏振预测
            target_12ch: [B,12,H,W] 或 tuple (None, target_12ch)
        Returns:
            total_loss, loss_dict
        """
        # 解包目标（兼容 tuple 传入）
        if isinstance(target_12ch, tuple):
            _, target_12ch = target_12ch

        # 从12通道目标提取 S0彩色 和 灰度偏振
        target_s0_rgb, target_gray_polar = self._extract_s0_rgb_and_gray_polar(target_12ch)

        # 解包预测
        pred_s0_rgb, pred_gray_polar = pred

        # 1. S0 L1 损失
        s0_loss = self.l1(pred_s0_rgb, target_s0_rgb)

        # 2. 计算预测和目标偏振参数
        pred_dolp, pred_aop = self._compute_polarization(pred_gray_polar)
        target_dolp, target_aop = self._compute_polarization(target_gray_polar)

        # 3. DoLP L1 损失
        dolp_loss = self.l1(pred_dolp, target_dolp)

        # 4. AoP 周期性 L1 损失
        aop_diff = torch.abs(pred_aop - target_aop)
        aop_diff = torch.min(aop_diff, math.pi - aop_diff)
        aop_loss = aop_diff.mean() / math.pi

        # 5. 物理一致性损失：I0 + I90 = I45 + I135
        intensity_loss = self.l1(pred_gray_polar, target_gray_polar)

        # 加权总loss
        total_loss = s0_loss +  dolp_loss +  aop_loss + 0.1 * intensity_loss

        loss_dict = {
            's0': s0_loss.item(),
            'dolp': dolp_loss.item(),
            'aop': aop_loss.item(),
            'physics': intensity_loss.item(),
            'total': total_loss.item()
        }
        return total_loss, loss_dict

class Trainer:
    def __init__(self, clean_root, noisy_root, config=None, use_amp=False,
                 pretrained_path=None, use_pt=True, normalize_brightness=False,
                 target_brightness=0.1, freeze_branch=None,
                 train_subdir="train", val_subdir="val",
                 save_path="best_model.pth"):
        """
        Args:
            clean_root: 清晰图像根目录（包含 train/ 和 test/ 子目录）
            noisy_root: 噪声图像根目录（若 use_pt=True 可留空）
            config: 配置字典（可选）
            use_amp: 是否使用混合精度
            pretrained_path: 预训练模型路径
            use_pt: 是否使用预处理好的 .pt 文件
            normalize_brightness: 是否对输入进行亮度归一化
            target_brightness: 归一化目标亮度
            freeze_branch: 冻结分支 ('s0', 'polar', None)
            train_subdir: 训练集子目录名（默认 "train"）
            val_subdir: 验证集子目录名（默认 "val"）
            save_path: 最优模型保存路径
        """
        self.use_pt = use_pt
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.model_path = save_path
        self.freeze_branch = freeze_branch

        # 创建模型（注意：模型输入通道必须为12）
        self.model = LPDFlash().to(self.device)  # LPDFlash 默认 use_12ch_input=True，接受 12 通道输入
        print("[INFO] 使用 LPDFlash (彩色偏振双分支去噪网络，12通道输入)")
        print("[INFO] 分支1: S0彩色图像 (3通道)")
        print("[INFO] 分支2: 灰度偏振图像 (4通道)")

        # 加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_pretrained(pretrained_path)
        elif os.path.exists(self.model_path):
            self.load_pretrained(self.model_path)
        else:
            print("[INFO] 未找到预训练模型，从头开始训练")

        # 分支冻结
        self._apply_freeze()

        # 优化器
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)

        # 损失函数
        self.criterion = PolarizationLoss().to(self.device)

        # 混合精度
        if self.use_amp:
            self.scaler = torch.amp.GradScaler()
            print("[OK] 启用混合精度训练 (AMP)")
        else:
            self.scaler = None
            print("[INFO] 未启用混合精度训练")

        # 数据集
        self.train_dataset = BranchDataset(
            clean_dir=os.path.join(clean_root, train_subdir),
            noisy_dir=os.path.join(noisy_root, train_subdir) if noisy_root else "",
            mode='train',
            image_size=1024,
            crop_size=256,
            stride=256,
            use_pt=self.use_pt,
            normalize_brightness=normalize_brightness,
            target_brightness=target_brightness
        )
        self.val_dataset = BranchDataset(
            clean_dir=os.path.join(clean_root, val_subdir),
            noisy_dir=os.path.join(noisy_root, val_subdir) if noisy_root else "",
            mode='val',
            image_size=1024,
            crop_size=256,
            stride=256,
            use_pt=self.use_pt,
            normalize_brightness=normalize_brightness,
            target_brightness=target_brightness
        )
        print(f"训练集: {len(self.train_dataset)}, 验证集: {len(self.val_dataset)}")

    # ---------- 辅助函数 ----------
    def _extract_target_from_12ch(self, target_12ch):
        """从12通道提取S0彩色和灰度偏振（与损失函数一致）"""
        B, C, H, W = target_12ch.shape
        polar_rgb = target_12ch.view(B, 4, 3, H, W)
        s0_rgb = polar_rgb.mean(dim=1)
        weights = torch.tensor([0.299, 0.587, 0.114], device=target_12ch.device).view(1, 3, 1, 1)
        gray_polar = (polar_rgb * weights).sum(dim=2)
        return s0_rgb, gray_polar

    def compute_polarization_parameters(self, gray_polar):
        """从4通道灰度偏振图像计算 S0(灰度), DoLP, AoP（用于验证指标）"""
        eps = 1e-8
        i0 = gray_polar[:, 0:1, :, :]
        i45 = gray_polar[:, 1:2, :, :]
        i90 = gray_polar[:, 2:3, :, :]
        i135 = gray_polar[:, 3:4, :, :]
        S0 = (i0 + i45 + i90 + i135) * 0.25
        S1 = (i0 - i90) * 0.5
        S2 = (i45 - i135) * 0.5
        DoLP = torch.sqrt(S1 ** 2 + S2 ** 2 + eps) / (S0 + eps)
        DoLP = torch.clamp(DoLP, 0.0, 1.0)
        AoP = 0.5 * torch.atan2(S2, S1 + eps)
        AoP = AoP % math.pi
        return S0, DoLP, AoP

    # ---------- 模型加载/保存 ----------
    def load_pretrained(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            if all(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[OK] 加载模型: {model_path}")
        except Exception as e:
            print(f"[FAIL] 加载失败: {e}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[OK] 模型已保存: {path}")

    # ---------- 分支冻结 ----------
    def _apply_freeze(self):
        if self.freeze_branch is None:
            print("[INFO] 训练所有分支")
            return
        if self.freeze_branch == 's0':
            self.model.set_freeze('s0')
            print("[INFO] 冻结S0分支，只训练偏振分支")
        elif self.freeze_branch == 'polar':
            self.model.set_freeze('polar')
            print("[INFO] 冻结偏振分支，只训练S0分支")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] 可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def unfreeze_all(self):
        self.model.set_freeze(None)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        print("[INFO] 已解冻所有分支")

    def freeze_s0_branch(self):
        self.model.set_freeze('s0')
        print("[INFO] 已冻结S0分支")

    def freeze_polar_branch(self):
        self.model.set_freeze('polar')
        print("[INFO] 已冻结偏振分支")

    # ---------- 训练主循环 ----------
    def train(self, epochs, resume_epoch=0):
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        best_loss = float('inf')
        early_stop_counter = 0
        accumulation_steps = 1

        for epoch in range(resume_epoch, resume_epoch + epochs):
            self.model.train()
            train_loss = 0.0

            for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
                noisy_12ch, clean_12ch = batch_data  # [B,12,H,W]
                noisy_12ch = noisy_12ch.to(self.device)
                clean_12ch = clean_12ch.to(self.device)

                # 前向传播
                branch1_out, branch2_out = self.model(noisy_12ch)
                output = (branch1_out, branch2_out)

                # 计算损失
                loss, _ = self.criterion(output, clean_12ch)
                loss = loss / accumulation_steps

                # 反向传播
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss += loss.item() * accumulation_steps

            # ---------- 验证 ----------
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_data in val_loader:
                    noisy_12ch, clean_12ch = batch_data
                    noisy_12ch = noisy_12ch.to(self.device)
                    clean_12ch = clean_12ch.to(self.device)

                    branch1_out, branch2_out = self.model(noisy_12ch)
                    output = (branch1_out, branch2_out)
                    loss, _ = self.criterion(output, clean_12ch)
                    val_loss += loss.item()

            # 平均
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 简化的打印信息
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                early_stop_counter = 0
                self.save_model(self.model_path)
                print(f"  [SAVED] 最佳模型 (Val Loss: {avg_val_loss:.6f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= 100:
                    print(f"早停于 epoch {epoch+1}")
                    break

def main():
    parser = argparse.ArgumentParser(description="LPDFlash 训练脚本")
    parser.add_argument("--clean_root", type=str, required=True,
                        help="清晰/噪声数据根目录（包含 train/、val/ 等子目录，内含 12 通道 .pt 或 0/45/90/135.png）")
    parser.add_argument("--noisy_root", type=str, default="",
                        help="噪声图像根目录；使用 .pt 文件时留空（.pt 内含 clean+noisy）")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--use_pt", action="store_true", default=True,
                        help="使用预处理好的 .pt 文件（默认启用）")
    parser.add_argument("--no_pt", action="store_false", dest="use_pt",
                        help="改用原始 0/45/90/135.png 图片")
    parser.add_argument("--freeze_branch", type=str, default=None,
                        choices=["s0", "polar"],
                        help="冻结分支：'s0' 或 'polar'；缺省训练全部")
    parser.add_argument("--train_subdir", type=str, default="train",
                        help="训练集子目录名（默认 train）")
    parser.add_argument("--val_subdir", type=str, default="val",
                        help="验证集子目录名（默认 val）")
    parser.add_argument("--save_path", type=str, default="best_model.pth",
                        help="最优模型保存路径")
    args = parser.parse_args()

    trainer = Trainer(
        args.clean_root,
        args.noisy_root,
        use_pt=args.use_pt,
        normalize_brightness=False,
        freeze_branch=args.freeze_branch,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        save_path=args.save_path,
    )

    trainer.train(args.epochs)


if __name__ == "__main__":
    main()