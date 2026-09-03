"""
LPDFlash 推理测试脚本 - 12通道输入格式
输入: 4方向×RGB = 12通道
输出: S0彩色(3通道) + 灰度偏振(4通道)
"""
import os
import json
import math
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import kornia
from models.LPDFlash import LPDFlash



def load_image(file_path, image_size=1024, mode='color', center_crop=True):
    """加载并预处理单张图像（彩色或灰度）"""
    if mode == 'color':
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法读取图像: {file_path}")

        if center_crop:
            h, w = img.shape[:2]
            if h > image_size or w > image_size:
                start_y = (h - image_size) // 2
                start_x = (w - image_size) // 2
                img = img[start_y:start_y + image_size, start_x:start_x + image_size, :]

        if img.shape[0] != image_size or img.shape[1] != image_size:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # [3,H,W]
    else:  # gray
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {file_path}")

        if center_crop:
            h, w = img.shape
            if h > image_size or w > image_size:
                start_y = (h - image_size) // 2
                start_x = (w - image_size) // 2
                img = img[start_y:start_y + image_size, start_x:start_x + image_size]

        if img.shape[0] != image_size or img.shape[1] != image_size:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

        tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)  # [1,H,W]

    return tensor


def load_12ch_sample(folder_path, image_size=1024, center_crop=True):
    """
    加载12通道样本：4张彩色图 (0°,45°,90°,135°) 拼接为 [12,H,W]
    顺序: R0,G0,B0, R45,G45,B45, R90,G90,B90, R135,G135,B135
    """
    angle_files = ["0.bmp", "45.bmp", "90.bmp", "135.bmp"]
    tensors = []
    for f in angle_files:
        file_path = os.path.join(folder_path, f)
        color_t = load_image(file_path, image_size, mode='color', center_crop=center_crop)  # [3,H,W]
        tensors.append(color_t)
    combined = torch.cat(tensors, dim=0)  # [12,H,W]
    return combined.unsqueeze(0)  # [1,12,H,W]


def normalize_brightness_12ch(input_tensor, target_mean=0.06, eps=1e-8):
    """亮度归一化（只对前3通道（0° RGB）处理，保持颜色比例）"""
    s0_channels = input_tensor[:, 0:3, :, :]  # 取0°方向的RGB作为S0近似
    current_mean = s0_channels.mean()
    if current_mean > eps:
        scale = target_mean / current_mean
        normalized = input_tensor.clone()
        normalized[:, 0:3, :, :] = torch.clamp(s0_channels * scale, 0.0, 1.0)
    else:
        normalized = input_tensor
    return normalized, scale


def compute_polarization_from_4ch(polar_gray):
    """
    从4通道灰度偏振计算偏振参数
    Args:
        polar_gray: [B, 4, H, W] - (I0, I45, I90, I135)
    Returns:
        S0, DoLP, AoP
    """
    eps = 1e-8
    i0 = polar_gray[:, 0:1, :, :]
    i45 = polar_gray[:, 1:2, :, :]
    i90 = polar_gray[:, 2:3, :, :]
    i135 = polar_gray[:, 3:4, :, :]

    s0 = (i0 + i45 + i90 + i135) * 0.25
    s1 = (i0 - i90) * 0.5
    s2 = (i45 - i135) * 0.5

    dolp = torch.sqrt(s1 ** 2 + s2 ** 2 + eps) / (s0 + eps)
    dolp = torch.clamp(dolp, 0.0, 1.0)

    aop = 0.5 * torch.atan2(s2, s1 + eps)
    aop = aop % math.pi
    return s0, dolp, aop


def calculate_metrics(pred_s0, pred_dolp, pred_aop, clean_s0, clean_dolp, clean_aop,
                      noisy_s0=None, noisy_dolp=None, noisy_aop=None):
    """计算评价指标"""
    metrics = {}

    # S0 指标
    metrics['s0_psnr'] = kornia.metrics.psnr(pred_s0, clean_s0, max_val=1.0).item()
    metrics['s0_ssim'] = kornia.metrics.ssim(pred_s0, clean_s0, window_size=11, max_val=1.0).mean().item()

    # DoLP 指标
    metrics['dolp_psnr'] = kornia.metrics.psnr(pred_dolp, clean_dolp, max_val=1.0).item()
    metrics['dolp_ssim'] = kornia.metrics.ssim(pred_dolp, clean_dolp, window_size=11, max_val=1.0).mean().item()
    metrics['dolp_mae'] = torch.abs(pred_dolp - clean_dolp).mean().item()

    # AoP 指标（周期性）
    aop_diff = torch.abs(pred_aop - clean_aop)
    aop_diff = torch.minimum(aop_diff, torch.tensor(math.pi, device=aop_diff.device) - aop_diff)
    metrics['aop_mae'] = aop_diff.mean().item()

    aop_mse = (aop_diff ** 2).mean()
    metrics['aop_psnr'] = 10 * math.log10((math.pi ** 2) / aop_mse.item()) if aop_mse > 0 else 100.0

    # AoP SSIM（直接对归一化角度计算，忽略周期性）
    pred_aop_norm = pred_aop / math.pi
    clean_aop_norm = clean_aop / math.pi
    metrics['aop_ssim'] = kornia.metrics.ssim(pred_aop_norm, clean_aop_norm, window_size=11, max_val=1.0).mean().item()

    # 噪声图像指标
    if noisy_s0 is not None:
        metrics['noisy_s0_psnr'] = kornia.metrics.psnr(noisy_s0, clean_s0, max_val=1.0).item()
        metrics['noisy_s0_ssim'] = kornia.metrics.ssim(noisy_s0, clean_s0, window_size=11, max_val=1.0).mean().item()

    if noisy_dolp is not None:
        metrics['noisy_dolp_psnr'] = kornia.metrics.psnr(noisy_dolp, clean_dolp, max_val=1.0).item()
        metrics['noisy_dolp_ssim'] = kornia.metrics.ssim(noisy_dolp, clean_dolp, window_size=11, max_val=1.0).mean().item()
        metrics['noisy_dolp_mae'] = torch.abs(noisy_dolp - clean_dolp).mean().item()

    if noisy_aop is not None:
        noisy_aop_diff = torch.abs(noisy_aop - clean_aop)
        noisy_aop_diff = torch.minimum(noisy_aop_diff, torch.tensor(math.pi, device=noisy_aop_diff.device) - noisy_aop_diff)
        metrics['noisy_aop_mae'] = noisy_aop_diff.mean().item()

        noisy_aop_mse = (noisy_aop_diff ** 2).mean()
        metrics['noisy_aop_psnr'] = 10 * math.log10((math.pi ** 2) / noisy_aop_mse.item()) if noisy_aop_mse > 0 else 100.0

        noisy_aop_norm = noisy_aop / math.pi
        metrics['noisy_aop_ssim'] = kornia.metrics.ssim(noisy_aop_norm, clean_aop_norm, window_size=11, max_val=1.0).mean().item()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LPDFlash 推理/测试脚本")
    parser.add_argument("--clean_root", type=str, required=True,
                        help="清晰(真值)样本根目录，内含样本文件夹(0/45/90/135.bmp)或 .pt 文件")
    parser.add_argument("--noisy_root", type=str, default=None,
                        help="噪声样本根目录（文件夹模式时使用）")
    parser.add_argument("--model_path", type=str, default="best_model.pth",
                        help="模型权重路径")
    parser.add_argument("--output_dir", type=str, default="results_test",
                        help="结果输出目录")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--no_center_crop", action="store_false", dest="center_crop",
                        help="禁用中心裁剪")
    parser.add_argument("--use_iap", action="store_true", default=False,
                        help="启用模型内部 IAP 预处理（默认关闭，与权重训练配置保持一致）")
    args = parser.parse_args()

    # ===== 配置 =====
    CLEAN_ROOT = args.clean_root
    NOISY_ROOT = args.noisy_root
    MODEL_PATH = args.model_path
    OUTPUT_DIR = args.output_dir
    IMAGE_SIZE = args.image_size
    CENTER_CROP = args.center_crop

    # 亮度归一化（外部预处理，建议关闭，模型内部已有 IAP）
    NORMALIZE_BRIGHTNESS = False
    TARGET_BRIGHTNESS = 0.28

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型（12通道输入）
    print("加载模型...")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 模型文件不存在: {MODEL_PATH}")
        return

    # 创建模型（使用12通道输入，IAP 开关与权重训练配置保持一致）
    model_train = LPDFlash(base_ch=8, rep_scale=8, iap_enabled=args.use_iap)

    # 加载权重
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 尝试加载（严格模式，以防权重不匹配）
    try:
        model_train.load_state_dict(state_dict, strict=True)
        print("[OK] 模型权重加载成功")
    except Exception as e:
        print(f"[WARN] 严格加载失败，尝试非严格加载: {e}")
        model_train.load_state_dict(state_dict, strict=False)
        print("[OK] 模型权重加载（非严格）")

    # 转换为推理版（slim）
    if hasattr(model_train, 'slim'):
        model = model_train.slim().to(device)
        print("[OK] 已转换为推理版（slim）")
    else:
        model = model_train.to(device)
        print("[INFO] 直接使用训练版推理")

    model.eval()

    # 获取测试样本
    if not os.path.exists(CLEAN_ROOT):
        print(f"[ERROR] 数据路径不存在: {CLEAN_ROOT}")
        return

    # 获取所有测试样本（支持文件夹或.pt文件）
    test_items = []
    # 先检查是否有文件夹
    folders = sorted([f for f in os.listdir(CLEAN_ROOT) if os.path.isdir(os.path.join(CLEAN_ROOT, f))])
    if folders:
        test_items = folders
        if not NOISY_ROOT or not os.path.exists(NOISY_ROOT):
            print("[ERROR] 文件夹模式需要 --noisy_root 指定噪声样本目录")
            return
    else:
        # 查找.pt文件
        pt_files = sorted([f for f in os.listdir(CLEAN_ROOT) if f.endswith('.pt')])
        if pt_files:
            test_items = pt_files
        else:
            print("[ERROR] 未找到任何测试样本（文件夹或.pt文件）")
            return

    print(f"找到 {len(test_items)} 个测试样本")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_metrics = []

    with torch.no_grad():
        for item in tqdm(test_items, desc="测试中"):
            try:
                # 判断是文件夹还是pt文件
                if item.endswith('.pt'):
                    # 加载 .pt 文件（期望是12通道）
                    pt_path = os.path.join(CLEAN_ROOT, item)
                    data = torch.load(pt_path, map_location=device)
                    if isinstance(data, dict):
                        # 读取 12 通道数据（clean + noisy）
                        if 'noisy' in data and data['noisy'].shape[0] == 12:
                            noisy_12ch = data['noisy'].unsqueeze(0).to(device)
                            clean_12ch = data['clean'].unsqueeze(0).to(device) if 'clean' in data else noisy_12ch
                        else:
                            # 旧7通道数据，转换为12通道？但无法完全恢复，警告并跳过
                            print(f"[WARN] {item} 不是12通道格式，跳过")
                            continue
                    else:
                        # 假设是 tensor [12, H, W]
                        if data.shape[0] == 12:
                            noisy_12ch = data.unsqueeze(0).to(device)
                            clean_12ch = noisy_12ch
                        else:
                            print(f"[WARN] {item} 通道数不为12，跳过")
                            continue
                else:
                    # 文件夹模式
                    noisy_path = os.path.join(NOISY_ROOT, item)
                    clean_path = os.path.join(CLEAN_ROOT, item)
                    noisy_12ch = load_12ch_sample(noisy_path, IMAGE_SIZE, center_crop=CENTER_CROP).to(device)
                    clean_12ch = load_12ch_sample(clean_path, IMAGE_SIZE, center_crop=CENTER_CROP).to(device)

                # 可选外部亮度归一化（对前3通道）
                if NORMALIZE_BRIGHTNESS:
                    noisy_12ch, _ = normalize_brightness_12ch(noisy_12ch, target_mean=TARGET_BRIGHTNESS)

                # 推理（模型内部自动将 12 通道拆分为 S0(3ch) 与灰度偏振(4ch)）
                branch1_out, branch2_out = model(noisy_12ch)
                # branch1_out: [B,3,H,W] S0彩色
                # branch2_out: [B,4,H,W] 灰度偏振

                # 计算预测的S0灰度（用于指标）
                pred_s0_color = branch1_out
                pred_s0_gray = pred_s0_color.mean(dim=1, keepdim=True)  # [B,1,H,W]

                # 从branch2_out计算DoLP, AoP
                _, pred_dolp, pred_aop = compute_polarization_from_4ch(branch2_out)

                # 获取真值的S0灰度、DoLP、AoP
                # 需要从12通道clean中提取灰度偏振，然后计算
                # 先提取12通道中的灰度偏振：4方向×RGB -> 转灰度
                # 利用模型内部的转换方法（但这里手动提取）
                B, C, H, W = clean_12ch.shape
                # 重排为 [B,4,3,H,W]
                polar_rgb = clean_12ch.view(B, 4, 3, H, W)
                # 转灰度（使用标准权重 RGB 顺序）
                rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=clean_12ch.device).view(1, 3, 1, 1)
                clean_polar_gray = (polar_rgb * rgb_weights).sum(dim=2)  # [B,4,H,W]
                # 计算真值S0彩色（用于保存）和偏振参数
                clean_s0_rgb = polar_rgb.mean(dim=1)  # [B,3,H,W]
                clean_s0_gray = clean_s0_rgb.mean(dim=1, keepdim=True)
                _, clean_dolp, clean_aop = compute_polarization_from_4ch(clean_polar_gray)

                # 噪声图像的偏振参数（用于指标）
                noisy_polar_rgb = noisy_12ch.view(B, 4, 3, H, W)
                noisy_polar_gray = (noisy_polar_rgb * rgb_weights).sum(dim=2)
                _, noisy_dolp, noisy_aop = compute_polarization_from_4ch(noisy_polar_gray)
                noisy_s0_rgb = noisy_polar_rgb.mean(dim=1)
                noisy_s0_gray = noisy_s0_rgb.mean(dim=1, keepdim=True)

                # 计算指标
                metrics = calculate_metrics(
                    pred_s0_gray, pred_dolp, pred_aop,
                    clean_s0_gray, clean_dolp, clean_aop,
                    noisy_s0_gray, noisy_dolp, noisy_aop,
                )
                metrics['sample'] = item
                all_metrics.append(metrics)

                # ===== 保存结果图片 =====
                sample_dir = os.path.join(OUTPUT_DIR, item.replace('.pt', ''))
                os.makedirs(sample_dir, exist_ok=True)

                def save_image(tensor, name, is_dolp=False, is_aop=False):
                    arr = tensor[0, 0].cpu().numpy() if tensor.dim() == 4 else tensor[0].cpu().numpy()
                    if is_aop:
                        arr = (arr / math.pi * 180).clip(0, 180).astype(np.uint8)
                        hsv = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
                        hsv[:, :, 0] = (arr / 180 * 179).astype(np.uint8)
                        hsv[:, :, 1] = 255
                        hsv[:, :, 2] = 255
                        img_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    elif is_dolp:
                        arr = np.clip(np.power(arr, 0.5) * 255, 0, 255).astype(np.uint8)
                        img_bgr = cv2.applyColorMap(arr, cv2.COLORMAP_VIRIDIS)
                    else:
                        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                        img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                    cv2.imwrite(os.path.join(sample_dir, f"{name}.png"), img_bgr)

                def save_color_image(tensor, name):
                    arr = tensor[0].cpu().numpy()  # [3,H,W]
                    arr = np.transpose(arr, (1, 2, 0))
                    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(sample_dir, f"{name}.png"), img_bgr)

                # 保存S0彩色
                save_color_image(pred_s0_color, "pred_S0_color")
                save_color_image(clean_s0_rgb, "clean_S0_color")
                save_color_image(noisy_s0_rgb, "noisy_S0_color")

                # 保存S0灰度
                save_image(pred_s0_gray, "pred_S0_gray")
                save_image(clean_s0_gray, "clean_S0_gray")
                save_image(noisy_s0_gray, "noisy_S0_gray")

                # 保存DoLP
                save_image(pred_dolp, "pred_DoLP", is_dolp=True)
                save_image(clean_dolp, "clean_DoLP", is_dolp=True)
                save_image(noisy_dolp, "noisy_DoLP", is_dolp=True)

                # 保存AoP
                save_image(pred_aop, "pred_AoP", is_aop=True)
                save_image(clean_aop, "clean_AoP", is_aop=True)
                save_image(noisy_aop, "noisy_AoP", is_aop=True)

                # ===== 保存分支2的四方向灰度图像 =====
                angles = ["0", "45", "90", "135"]
                for idx, angle in enumerate(angles):
                    # 预测的四方向灰度
                    pred_polar = branch2_out[:, idx:idx+1, :, :]
                    save_image(pred_polar, f"pred_polar_{angle}")
                    # 真值的四方向灰度
                    clean_polar = clean_polar_gray[:, idx:idx+1, :, :]
                    save_image(clean_polar, f"clean_polar_{angle}")
                    # 噪声的四方向灰度
                    noisy_polar = noisy_polar_gray[:, idx:idx+1, :, :]
                    save_image(noisy_polar, f"noisy_polar_{angle}")

            except Exception as e:
                print(f"\n[ERROR] 处理 {item} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 汇总结果
    if all_metrics:
        avg_metrics = {}
        base_keys = ['s0_psnr', 's0_ssim', 'dolp_psnr', 'dolp_ssim', 'aop_psnr', 'aop_ssim', 'aop_mae']
        if 'dolp_mae' in all_metrics[0]:
            base_keys.append('dolp_mae')
        for key in base_keys:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        noisy_keys = ['noisy_s0_psnr', 'noisy_s0_ssim', 'noisy_dolp_psnr', 'noisy_dolp_ssim',
                      'noisy_aop_psnr', 'noisy_aop_ssim', 'noisy_aop_mae']
        if 'noisy_dolp_mae' in all_metrics[0]:
            noisy_keys.append('noisy_dolp_mae')
        for key in noisy_keys:
            if key in all_metrics[0]:
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        summary = {
            'total_samples': len(all_metrics),
            'average_metrics': avg_metrics,
            'per_sample_metrics': all_metrics,
            'config': {
                'normalize_brightness': NORMALIZE_BRIGHTNESS,
                'target_brightness': TARGET_BRIGHTNESS if NORMALIZE_BRIGHTNESS else None,
            }
        }

        with open(os.path.join(OUTPUT_DIR, "test_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

        print("\n" + "=" * 60)
        print("测试结果汇总（12通道输入）")
        print("=" * 60)
        print(f"总样本数: {len(all_metrics)}")
        print(f"亮度归一化: {'启用' if NORMALIZE_BRIGHTNESS else '禁用'}")
        print("-" * 60)
        print("【预测结果 vs 真值】")
        s0_str = f"  S0:   PSNR={avg_metrics['s0_psnr']:.2f}dB, SSIM={avg_metrics['s0_ssim']:.4f}"
        print(s0_str)
        dolp_str = f"  DoLP: PSNR={avg_metrics['dolp_psnr']:.2f}dB, SSIM={avg_metrics['dolp_ssim']:.4f}"
        if 'dolp_mae' in avg_metrics:
            dolp_str += f", MAE={avg_metrics['dolp_mae']:.4f}"
        print(dolp_str)
        print(f"  AoP:  PSNR={avg_metrics['aop_psnr']:.2f}dB, SSIM={avg_metrics['aop_ssim']:.4f}, MAE={avg_metrics['aop_mae']:.4f}rad")
        print("-" * 60)

        if 'noisy_s0_psnr' in avg_metrics:
            print("【噪声图像 vs 真值】")
            noisy_s0_str = f"  S0:   PSNR={avg_metrics['noisy_s0_psnr']:.2f}dB, SSIM={avg_metrics['noisy_s0_ssim']:.4f}"
            print(noisy_s0_str)
            noisy_dolp_str = f"  DoLP: PSNR={avg_metrics['noisy_dolp_psnr']:.2f}dB, SSIM={avg_metrics['noisy_dolp_ssim']:.4f}"
            if 'noisy_dolp_mae' in avg_metrics:
                noisy_dolp_str += f", MAE={avg_metrics['noisy_dolp_mae']:.4f}"
            print(noisy_dolp_str)
            print(f"  AoP:  PSNR={avg_metrics['noisy_aop_psnr']:.2f}dB, SSIM={avg_metrics['noisy_aop_ssim']:.4f}, MAE={avg_metrics['noisy_aop_mae']:.4f}rad")
            print("-" * 60)
            print("【提升幅度】")
            s0_improve = f"  S0:   PSNR +{avg_metrics['s0_psnr'] - avg_metrics['noisy_s0_psnr']:.2f}dB"
            print(s0_improve)
            dolp_improve = f"  DoLP: PSNR +{avg_metrics['dolp_psnr'] - avg_metrics['noisy_dolp_psnr']:.2f}dB"
            if 'dolp_mae' in avg_metrics and 'noisy_dolp_mae' in avg_metrics:
                dolp_improve += f", MAE {avg_metrics['noisy_dolp_mae'] - avg_metrics['dolp_mae']:+.4f}"
            print(dolp_improve)
            print(f"  AoP:  PSNR +{avg_metrics['aop_psnr'] - avg_metrics['noisy_aop_psnr']:.2f}dB, MAE -{avg_metrics['noisy_aop_mae'] - avg_metrics['aop_mae']:.4f}rad")

        print(f"\n结果保存在: {OUTPUT_DIR}")

    print("\n[DONE] 测试完成!")


if __name__ == "__main__":
    main()