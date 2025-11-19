import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import lpips

# ✅ 仅保留所需扩展名支持（可选）
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']

# -----------------------------
# 工具函数：获取图像路径（保持简洁）
# -----------------------------
def get_image_paths(path, sort=False):
    import glob
    paths = []
    for ext in ALLOWED_IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(path, f'*.{ext}')))
    if sort:
        paths.sort()
    return paths

# -----------------------------
# 自定义 Dataset：加载图像
# -----------------------------
class ImagePathDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

# -----------------------------
# 数据加载器构建（用于 SSIM/PSNR/LPIPS 公共输入）
# -----------------------------
def load_image_loaders(content_paths, target_paths, batch_size=1, num_workers=0, img_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1] float
        transforms.Resize((img_size, img_size))
    ])
    content_loader = DataLoader(
        ImagePathDataset(content_paths, transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    target_loader = DataLoader(
        ImagePathDataset(target_paths, transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return content_loader, target_loader

# -----------------------------
# 计算 SSIM / PSNR（每对图像）
# -----------------------------
def compute_ssim_psnr(loader1, loader2, real_paths, fake_paths, metric='ssim'):
    assert len(loader1) == len(loader2), "Loaders must have same length"
    scores = []
    filenames = []

    real_basenames = [os.path.splitext(os.path.basename(p))[0] for p in real_paths]
    fake_basenames = [os.path.splitext(os.path.basename(p))[0] for p in fake_paths]

    pbar = tqdm(zip(loader1, loader2), total=len(loader1), desc=f"Computing {metric.upper()}")
    for idx, (batch1, batch2) in enumerate(pbar):
        for i in range(batch1.shape[0]):
            img1_np = (batch1[i].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            img2_np = (batch2[i].numpy() * 255).astype(np.uint8).transpose(1, 2, 0)

            if metric == 'ssim':
                score = ssim(img1_np, img2_np, channel_axis=2, data_range=255, multichannel=True)
            elif metric == 'psnr':
                score = psnr(img1_np, img2_np, data_range=255)
            else:
                raise ValueError("metric must be 'ssim' or 'psnr'")
            scores.append(score)
            filenames.append((
                real_basenames[idx * loader1.batch_size + i],
                fake_basenames[idx * loader1.batch_size + i]
            ))
        pbar.set_postfix({metric: f"{np.mean(scores[-batch1.shape[0]:]):.4f}"})
    return scores, filenames

# -----------------------------
# 初始化并计算 LPIPS（支持 alex/vgg；默认 alex）
# -----------------------------
def compute_lpips(loader1, loader2, net='alex', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = lpips.LPIPS(net=net).to(device).eval()
    scores = []

    pbar = tqdm(zip(loader1, loader2), total=len(loader1), desc="Computing LPIPS")
    for batch1, batch2 in pbar:
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        with torch.no_grad():
            dists = loss_fn(batch1, batch2)  # shape: [B, 1, 1, 1]
        scores.extend(dists.squeeze().cpu().tolist())
        pbar.set_postfix(lpips=f"{np.mean(scores[-batch1.shape[0]:]):.4f}")
    return scores

# -----------------------------
# 保存指标结果到 txt 文件
# -----------------------------
def save_metrics_to_txt(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== Summary ===\n")
        f.write(f"Mean SSIM: {results['ssim_mean']:.6f}\n")
        f.write(f"Mean PSNR: {results['psnr_mean']:.6f}\n")
        f.write(f"Mean LPIPS: {results['lpips_mean']:.6f}\n")
        f.write("\n=== Per-pair Metrics ===\n")
        f.write("Index, Real_Name, Fake_Name, SSIM, PSNR, LPIPS\n")
        for i, ((real_name, fake_name), ssim_val, psnr_val, lpips_val) in enumerate(
            zip(results['filenames'], results['ssim_scores'], results['psnr_scores'], results['lpips_scores'])
        ):
            f.write(f"{i+1}, {real_name}, {fake_name}, {ssim_val:.6f}, {psnr_val:.6f}, {lpips_val:.6f}\n")
    print(f"✅ Metrics saved to: {output_path}")

# -----------------------------
# 高层封装：一键计算三项指标 + 支持保存
# -----------------------------
def evaluate_ssim_psnr_lpips(real_dir, fake_dir, batch_size=32, num_workers=4, img_size=256, net='alex'):
    real_paths = sorted(get_image_paths(real_dir))
    fake_paths = sorted(get_image_paths(fake_dir))
    assert len(real_paths) == len(fake_paths), f"Length mismatch: {len(real_paths)} vs {len(fake_paths)}"

    loader_real, loader_fake = load_image_loaders(
        real_paths, fake_paths, batch_size=batch_size, num_workers=num_workers, img_size=img_size
    )

    ssim_scores, filenames = compute_ssim_psnr(loader_real, loader_fake, real_paths, fake_paths, metric='ssim')
    psnr_scores, _ = compute_ssim_psnr(loader_real, loader_fake, real_paths, fake_paths, metric='psnr')
    lpips_scores = compute_lpips(loader_real, loader_fake, net=net)

    return {
        'ssim_mean': np.mean(ssim_scores),
        'ssim_scores': ssim_scores,
        'psnr_mean': np.mean(psnr_scores),
        'psnr_scores': psnr_scores,
        'lpips_mean': np.mean(lpips_scores),
        'lpips_scores': lpips_scores,
        'filenames': filenames,
    }

# -----------------------------
# 示例用法（含保存）
# -----------------------------
if __name__ == '__main__':
    real_dir = "/media/data/github/BBDM-base/results/MIST-Ki67/LBBDM-f4-base-Ki67/sample_to_eval/ground_truth"
    fake_dir = "/media/data/github/BBDM-base/results/MIST-Ki67/LBBDM-f4-base-Ki67/sample_to_eval/200"
    output_txt = "/media/data/github/BBDM-base/results/Metrics_SSIM_PSNR_LPIPS.txt"

    results = evaluate_ssim_psnr_lpips(
        real_dir=real_dir,
        fake_dir=fake_dir,
        batch_size=16,
        num_workers=4,
        img_size=256,
        net='alex'
    )

    print("📊 Evaluation Results:")
    print(f"  SSIM  : {results['ssim_mean']:.6f}")
    print(f"  PSNR  : {results['psnr_mean']:.6f}")
    print(f"  LPIPS : {results['lpips_mean']:.6f}")

    # ✅ 保存到 txt
    save_metrics_to_txt(results, output_txt)