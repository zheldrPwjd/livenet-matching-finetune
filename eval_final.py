import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob

sys.path.append(r'C:\Users\user\Desktop\17\repos\LIVENET\src')
from models.network import get_model
from utils import parse, select_device

sys.path.append(r'C:\Users\user\Desktop\17\repos\LOFTR')
from src.loftr import LoFTR, default_cfg

ANGLES   = [10, 15, 20, 25, 30]
EVAL_LOW = r'C:\Users\user\Desktop\17\repos\LIVENET\data\eval15\low'
OPT_PATH = r'C:\Users\user\Desktop\17\repos\LIVENET\src\cfs\lolv1.yaml'
CKP      = r'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints'

FINAL_GEN = fr'{CKP}\ft_final_conf10.0_sob0.5_perc0.5_fft0.1\gen_epoch149.pth'
FINAL_REF = fr'{CKP}\ft_final_conf10.0_sob0.5_perc0.5_fft0.1\ref_epoch149.pth'

def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)
    return model

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def compute_inlier_ratio(g0, g1, matcher, device):
    t0 = torch.from_numpy(g0/255.).float().unsqueeze(0).unsqueeze(0).to(device)
    t1 = torch.from_numpy(g1/255.).float().unsqueeze(0).unsqueeze(0).to(device)
    t0 = F.interpolate(t0, size=(480,640))
    t1 = F.interpolate(t1, size=(480,640))
    batch = {'image0': t0, 'image1': t1}
    with torch.no_grad():
        matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    if len(mkpts0) < 4:
        return 0.0
    _, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 3.0)
    if mask is None:
        return 0.0
    return float(mask.sum()) / len(mask)

def enhance_livenet(img_rgb, generator, refiner, device):
    t = torch.from_numpy(img_rgb/255.).float().permute(2,0,1).unsqueeze(0).to(device)
    t512 = F.interpolate(t, size=(512,512))
    with torch.no_grad():
        pred_gray, _, _, pred_refined = generator(t512)
        pred_rgb = refiner(pred_refined, pred_gray)
    pred_rgb = F.interpolate(pred_rgb, size=(img_rgb.shape[0], img_rgb.shape[1]))
    out = pred_rgb.squeeze(0).permute(1,2,0).cpu().numpy()
    return np.clip(out*255, 0, 255).astype(np.uint8)

def main():
    opt = parse(OPT_PATH)
    device = select_device()
    opt['device'] = device

    matcher = LoFTR(config=default_cfg)
    ckpt = torch.load(r'C:\Users\user\Desktop\17\repos\LOFTR\weights\indoor_ds_new.ckpt')
    matcher.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    matcher = matcher.eval().to(device)

    low_imgs = sorted(glob(os.path.join(EVAL_LOW, '*.png')))
    print(f"eval15: {len(low_imgs)}장")
    print(f"ft_final gen: {FINAL_GEN}\n")

    generator, refiner = get_model(opt['model'], device=device)
    generator = load_weights(generator, FINAL_GEN, device)
    refiner   = load_weights(refiner,   FINAL_REF, device)
    generator.eval(); refiner.eval()
    print("ft_final 가중치 로드 완료!\n")

    results = {a: [] for a in ANGLES}

    for low_path in low_imgs:
        low_rgb  = cv2.cvtColor(cv2.imread(low_path), cv2.COLOR_BGR2RGB)
        enh_gray = cv2.cvtColor(enhance_livenet(low_rgb, generator, refiner, device), cv2.COLOR_RGB2GRAY)
        for angle in ANGLES:
            results[angle].append(
                compute_inlier_ratio(enh_gray, rotate_image(enh_gray, angle), matcher, device))

    print("===== ft_final 결과 =====")
    for a in ANGLES:
        print(f"  {a}도: {np.mean(results[a]):.4f}")

    # 기존 전체 비교
    prev = {
        'original':       {10:0.855, 15:0.728, 20:0.378, 25:0.067, 30:0.000},
        'zerodce':        {10:0.856, 15:0.790, 20:0.697, 25:0.633, 30:0.509},
        'base':           {10:0.851, 15:0.776, 20:0.666, 25:0.596, 30:0.532},
        'ft_lam10':       {10:0.849, 15:0.771, 20:0.670, 25:0.606, 30:0.508},
        'ft_sobel':       {10:0.854, 15:0.790, 20:0.684, 25:0.613, 30:0.588},
        'ft_sobel_perc':  {10:0.857, 15:0.790, 20:0.685, 25:0.591, 30:0.548},
        'ft_hf':          {10:0.850, 15:0.790, 20:0.649, 25:0.574, 30:0.611},
    }

    print(f"\n{'각도':<6}", end="")
    for m in list(prev.keys()) + ['ft_final']:
        print(f"{m:<16}", end="")
    print()

    for a in ANGLES:
        print(f"{a}도    ", end="")
        for m in prev:
            print(f"{prev[m][a]:.4f}          ", end="")
        print(f"{np.mean(results[a]):.4f}          ", end="")
        print()

    print("\n===== ft_final vs 각 모델 =====")
    for m in prev:
        print(f"vs {m}:")
        for a in ANGLES:
            diff = np.mean(results[a]) - prev[m][a]
            print(f"  {a}도: {diff:+.4f}")
        print()

    print("평가 완료!")

if __name__ == "__main__":
    main()
