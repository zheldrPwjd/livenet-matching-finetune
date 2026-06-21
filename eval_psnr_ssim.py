import sys
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob

sys.path.append(r'C:\Users\user\Desktop\17\repos\LIVENET\src')
from models.network import get_model
from utils import parse, select_device

EVAL_LOW  = r'C:\Users\user\Desktop\17\repos\LIVENET\data\eval15\low'
EVAL_HIGH = r'C:\Users\user\Desktop\17\repos\LIVENET\data\eval15\high'
OPT_PATH  = r'C:\Users\user\Desktop\17\repos\LIVENET\src\cfs\lolv1.yaml'
CKP       = r'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints'

# ===== 모델별 체크포인트 경로 =====
MODELS = {
    'base': {
        'gen': fr'{CKP}\best_gen.pth',
        'ref': fr'{CKP}\best_refine.pth',
    },
    'ft_sobel': {
        'gen': fr'{CKP}\ft_sobel0.5\gen_epoch99.pth',
        'ref': fr'{CKP}\ft_sobel0.5\ref_epoch99.pth',
    },
    'ft_sobel_perc': {
        'gen': fr'{CKP}\ft_sobel_perc0.05\gen_epoch49.pth',
        'ref': fr'{CKP}\ft_sobel_perc0.05\ref_epoch49.pth',
    },
    'ft_final': {
        'gen': fr'{CKP}\ft_final_conf10.0_sob0.5_perc0.5_fft0.1\gen_epoch149.pth',
        'ref': fr'{CKP}\ft_final_conf10.0_sob0.5_perc0.5_fft0.1\ref_epoch149.pth',
    },
}


def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)
    return model


def enhance_livenet(img_rgb, generator, refiner, device):
    t = torch.from_numpy(img_rgb / 255.).float().permute(2, 0, 1).unsqueeze(0).to(device)
    t512 = F.interpolate(t, size=(512, 512))
    with torch.no_grad():
        pred_gray, _, _, pred_refined = generator(t512)
        pred_rgb = refiner(pred_refined, pred_gray)
    pred_rgb = F.interpolate(pred_rgb, size=(img_rgb.shape[0], img_rgb.shape[1]))
    out = pred_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(out * 255, 0, 255).astype(np.float32)


def compute_psnr(pred, gt):
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10((255.0 ** 2) / mse)


def gaussian_kernel(size=11, sigma=1.5):
    ax = np.arange(-(size // 2), size // 2 + 1)
    gauss = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()


def compute_ssim(pred, gt):
    # 단일 채널(그레이스케일)에 대한 SSIM, 11x11 가우시안 윈도우
    pred_g = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    gt_g = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)

    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    kernel = gaussian_kernel(11, 1.5)

    mu1 = cv2.filter2D(pred_g, -1, kernel)
    mu2 = cv2.filter2D(gt_g, -1, kernel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = cv2.filter2D(pred_g ** 2, -1, kernel) - mu1_sq
    sigma2_sq = cv2.filter2D(gt_g ** 2, -1, kernel) - mu2_sq
    sigma12 = cv2.filter2D(pred_g * gt_g, -1, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-models', nargs='+', default=['ft_sobel_perc', 'ft_final'],
                         help='평가할 모델 이름들 (MODELS 딕셔너리 키)')
    args = parser.parse_args()

    opt = parse(OPT_PATH)
    device = select_device()
    opt['device'] = device

    low_imgs = sorted(glob(os.path.join(EVAL_LOW, '*.png')))
    high_imgs = sorted(glob(os.path.join(EVAL_HIGH, '*.png')))
    print(f"eval15: low {len(low_imgs)}장, high {len(high_imgs)}장\n")

    for name in args.models:
        if name not in MODELS:
            print(f"[SKIP] '{name}' 은 MODELS에 정의되어 있지 않음")
            continue

        ckpt = MODELS[name]
        print(f"===== {name} =====")
        print(f"  gen: {ckpt['gen']}")
        print(f"  ref: {ckpt['ref']}")

        generator, refiner = get_model(opt['model'], device=device)
        generator = load_weights(generator, ckpt['gen'], device)
        refiner = load_weights(refiner, ckpt['ref'], device)
        generator.eval()
        refiner.eval()

        psnr_list, ssim_list = [], []
        for low_path, high_path in zip(low_imgs, high_imgs):
            low_rgb = cv2.cvtColor(cv2.imread(low_path), cv2.COLOR_BGR2RGB)
            high_rgb = cv2.cvtColor(cv2.imread(high_path), cv2.COLOR_BGR2RGB).astype(np.float32)

            pred_rgb = enhance_livenet(low_rgb, generator, refiner, device)
            if pred_rgb.shape[:2] != high_rgb.shape[:2]:
                pred_rgb = cv2.resize(pred_rgb, (high_rgb.shape[1], high_rgb.shape[0]))

            psnr_list.append(compute_psnr(pred_rgb, high_rgb))
            ssim_list.append(compute_ssim(pred_rgb, high_rgb))

        print(f"  PSNR: {np.mean(psnr_list):.2f} dB")
        print(f"  SSIM: {np.mean(ssim_list):.3f}")
        print()

    print("측정 완료!")


if __name__ == "__main__":
    main()
