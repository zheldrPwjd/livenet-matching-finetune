import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(r'C:\Users\user\Desktop\17\repos\LIVENET\src')
from models.network import get_model
from utils import parse, select_device

sys.path.append(r'C:\Users\user\Desktop\17\repos\LOFTR')
from src.loftr import LoFTR, default_cfg

NIGHT_DIR = r'C:\Users\user\Desktop\17\campus_night'
OPT_PATH  = r'C:\Users\user\Desktop\17\repos\LIVENET\src\cfs\lolv1.yaml'
CKP       = r'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints'

FINAL_GEN = fr'{CKP}\ft_final_conf10.0_sob0.5_perc0.5_fft0.1\gen_epoch149.pth'
FINAL_REF = fr'{CKP}\ft_final_conf10.0_sob0.5_perc0.5_fft0.1\ref_epoch149.pth'
HF_GEN    = fr'{CKP}\ft_hf_lam10.0_sob0.5_perc0.05\gen_epoch99.pth'
HF_REF    = fr'{CKP}\ft_hf_lam10.0_sob0.5_perc0.05\ref_epoch99.pth'

PLACES = [
    ('place1',  'easy'),
    ('place2',  'mid'),
    ('place3',  'hard'),
    ('place4',  'mid'),
    ('place5',  'mid'),
    ('place6',  'easy'),
    ('place7',  'hard'),
    ('place8',  'hard'),
    ('place9',  'hard'),
    ('place10', 'mid'),
    ('place11', 'easy'),
]

def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)
    return model

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

def to_gray_640(img_rgb):
    return cv2.cvtColor(cv2.resize(img_rgb, (640,480)), cv2.COLOR_RGB2GRAY)

def main():
    opt = parse(OPT_PATH)
    device = select_device()
    opt['device'] = device

    matcher = LoFTR(config=default_cfg)
    ckpt = torch.load(r'C:\Users\user\Desktop\17\repos\LOFTR\weights\indoor_ds_new.ckpt')
    matcher.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    matcher = matcher.eval().to(device)

    MODELS = {
        'ft_final': {'gen': FINAL_GEN, 'ref': FINAL_REF},
        'ft_hf':    {'gen': HF_GEN,    'ref': HF_REF},
    }

    results = {m: {'easy':[], 'mid':[], 'hard':[]} for m in MODELS}

    for model_name, ckpt_paths in MODELS.items():
        print(f"\n===== {model_name} =====")
        generator, refiner = get_model(opt['model'], device=device)
        generator = load_weights(generator, ckpt_paths['gen'], device)
        refiner   = load_weights(refiner,   ckpt_paths['ref'], device)
        generator.eval(); refiner.eval()

        for place, diff in PLACES:
            a = cv2.cvtColor(cv2.imread(os.path.join(NIGHT_DIR, f'{place}_a.jpg')), cv2.COLOR_BGR2RGB)
            b = cv2.cvtColor(cv2.imread(os.path.join(NIGHT_DIR, f'{place}_b.jpg')), cv2.COLOR_BGR2RGB)
            inlier = compute_inlier_ratio(
                to_gray_640(enhance_livenet(a, generator, refiner, device)),
                to_gray_640(enhance_livenet(b, generator, refiner, device)),
                matcher, device)
            results[model_name][diff].append(inlier)
            print(f"  {place} ({diff}): {inlier:.4f}")

    # 기존 결과 (참고용)
    prev = {
        'original':      {'easy':0.845, 'mid':0.556, 'hard':0.788, '전체':0.719},
        'zerodce':       {'easy':0.732, 'mid':0.570, 'hard':0.758, '전체':0.682},
        'base':          {'easy':0.699, 'mid':0.516, 'hard':0.770, '전체':0.658},
        'ft_lam10':      {'easy':0.738, 'mid':0.555, 'hard':0.779, '전체':0.686},
        'ft_sobel':      {'easy':0.765, 'mid':0.556, 'hard':0.761, '전체':0.688},
        'ft_sobel_perc': {'easy':0.805, 'mid':0.554, 'hard':0.801, '전체':0.712},
    }

    print("\n========== 난이도별 평균 Inlier (전체 모델) ==========")
    all_models = list(prev.keys()) + list(MODELS.keys())
    print(f"{'':8}", end="")
    for m in all_models:
        print(f"{m:<16}", end="")
    print()

    for diff in ['easy', 'mid', 'hard']:
        print(f"{diff}    ", end="")
        for m in prev:
            print(f"{prev[m][diff]:.4f}          ", end="")
        for m in MODELS:
            print(f"{np.mean(results[m][diff]):.4f}          ", end="")
        print()

    print(f"{'전체':8}", end="")
    for m in prev:
        print(f"{prev[m]['전체']:.4f}          ", end="")
    for m in MODELS:
        all_v = results[m]['easy'] + results[m]['mid'] + results[m]['hard']
        print(f"{np.mean(all_v):.4f}          ", end="")
    print()

    print("\n===== ft_final vs 각 모델 (전체) =====")
    final_all = results['ft_final']['easy'] + results['ft_final']['mid'] + results['ft_final']['hard']
    for m in prev:
        print(f"vs {m}: {np.mean(final_all)-prev[m]['전체']:+.4f}")
    hf_all = results['ft_hf']['easy'] + results['ft_hf']['mid'] + results['ft_hf']['hard']
    print(f"vs ft_hf: {np.mean(final_all)-np.mean(hf_all):+.4f}")

    print("\n평가 완료!")

if __name__ == "__main__":
    main()
