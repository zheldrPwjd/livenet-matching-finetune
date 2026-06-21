import argparse
import logging
import sys
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import lpips

sys.path.append(r'C:\Users\user\Desktop\17\repos\LOFTR')
from src.loftr import LoFTR, default_cfg

from models import get_model, GenLoss, RefineLoss
from utils import parse, select_device
from data import get_data


def sobel_xy_loss(pred, gt):
    """Sobel x, y 분리해서 각각 L1 (edge loss, 논문 검증 방식)"""
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=pred.device)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=pred.device)
    sx = sx.view(1,1,3,3).repeat(pred.shape[1],1,1,1)
    sy = sy.view(1,1,3,3).repeat(pred.shape[1],1,1,1)

    pred_gx = F.conv2d(pred, sx, padding=1, groups=pred.shape[1])
    pred_gy = F.conv2d(pred, sy, padding=1, groups=pred.shape[1])
    gt_gx   = F.conv2d(gt,   sx, padding=1, groups=gt.shape[1])
    gt_gy   = F.conv2d(gt,   sy, padding=1, groups=gt.shape[1])

    return F.l1_loss(pred_gx, gt_gx) + F.l1_loss(pred_gy, gt_gy)


def fft_loss(pred, gt):
    """주파수 도메인 손실 (NTIRE 2024 검증 방식)"""
    pred_fft = torch.fft.rfft2(pred, norm='backward')
    gt_fft   = torch.fft.rfft2(gt,   norm='backward')
    pred_mag = torch.abs(pred_fft)
    gt_mag   = torch.abs(gt_fft)
    return F.l1_loss(pred_mag, gt_mag)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt",      type=str,   required=True)
    parser.add_argument("-lam",      type=float, default=10.0)   # confidence
    parser.add_argument("-lam_sob",  type=float, default=0.5)    # sobel x,y
    parser.add_argument("-lam_perc", type=float, default=0.5)    # perceptual (NTIRE: 0.5)
    parser.add_argument("-lam_fft",  type=float, default=0.1)    # frequency (NTIRE: 0.1)
    parser.add_argument("-epochs",   type=int,   default=150)
    args = parser.parse_args()
    opt = parse(args.opt)
    opt["device"] = select_device()
    return opt, args


def main():
    opt, args = parse_config()
    lam, lam_sob, lam_perc, lam_fft = args.lam, args.lam_sob, args.lam_perc, args.lam_fft

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"FINAL FT | lam={lam}, lam_sob={lam_sob}, lam_perc={lam_perc}, lam_fft={lam_fft}, epochs={args.epochs}")

    train_dataloader, _ = get_data(opt)

    generator, refiner = get_model(opt['model'], device=opt['device'])

    # base 가중치 로드 (hf_branch/fusion은 strict=False로 랜덤 초기화)
    base_gen = r'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints\best_gen.pth'
    base_ref = r'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints\best_refine.pth'

    def load_w(model, path):
        ckpt = torch.load(path, map_location=opt['device'])
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt, strict=False)
        return model

    generator = load_w(generator, base_gen)
    refiner   = load_w(refiner,   base_ref)
    logger.info("base 가중치 로드 완료 (hf_branch/fusion 랜덤 초기화)")

    # 기존 레이어 lr 낮게, 새 hf_branch/fusion 레이어 lr 높게
    existing_params, new_params = [], []
    for name, param in generator.named_parameters():
        if 'hf_branch' in name or 'fusion' in name:
            new_params.append(param)
        else:
            existing_params.append(param)

    opt_gen = optim.Adam([
        {'params': existing_params, 'lr': 5e-6},
        {'params': new_params,      'lr': 1e-4},
    ])
    opt_ref = optim.Adam(refiner.parameters(), lr=5e-6)
    logger.info(f"기존 파라미터: {len(existing_params)}개 (lr=5e-6), 새 파라미터: {len(new_params)}개 (lr=1e-4)")

    gen_loss_fn = GenLoss()
    ref_loss_fn = RefineLoss()

    lpips_fn = lpips.LPIPS(net='vgg').to(opt['device'])
    for p in lpips_fn.parameters():
        p.requires_grad = False

    matcher = LoFTR(config=default_cfg)
    ckpt = torch.load(r'C:\Users\user\Desktop\17\repos\LOFTR\weights\indoor_ds_new.ckpt')
    matcher.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    matcher = matcher.eval().to(opt['device'])
    for p in matcher.parameters():
        p.requires_grad = False
    logger.info("LoFTR freeze done!")

    save_dir = fr'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints\ft_final_conf{lam}_sob{lam_sob}_perc{lam_perc}_fft{lam_fft}'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        generator.train()
        refiner.train()
        loop = tqdm(train_dataloader)
        total_conf = total_sob = total_perc = total_fft = 0.0

        for low_rgb, normal_rgb, normal_gray in loop:
            low_rgb     = low_rgb.to(opt['device'])
            normal_rgb  = normal_rgb.to(opt['device'])
            normal_gray = normal_gray.to(opt['device'])

            # ===== Generator step =====
            generator.zero_grad()
            pred_gray, pred_tm, pred_al, pred_refined = generator(low_rgb)
            g_loss = gen_loss_fn(normal_gray, pred_tm.detach(), pred_al.detach(),
                                 pred_gray, pred_tm, pred_al)

            pred_rgb_g = refiner(pred_refined, pred_gray)

            # Sobel x,y edge loss
            sob_loss = sobel_xy_loss(pred_rgb_g, normal_rgb)

            # Perceptual loss (LPIPS, [-1,1])
            pred_norm = pred_rgb_g * 2 - 1
            gt_norm   = normal_rgb * 2 - 1
            perc_loss = lpips_fn(pred_norm, gt_norm).mean()

            # FFT frequency loss
            f_loss = fft_loss(pred_rgb_g, normal_rgb)

            # LoFTR confidence loss
            gray0 = (0.299*pred_rgb_g[:,0] + 0.587*pred_rgb_g[:,1] + 0.114*pred_rgb_g[:,2]).unsqueeze(1)
            gray1 = (0.299*normal_rgb[:,0]  + 0.587*normal_rgb[:,1]  + 0.114*normal_rgb[:,2]).unsqueeze(1)
            gray0 = F.interpolate(gray0, size=(480,640))
            gray1 = F.interpolate(gray1, size=(480,640))
            batch_loftr = {'image0': gray0, 'image1': gray1}
            matcher(batch_loftr)
            conf_loss = -batch_loftr['mconf'].mean()

            total_g = (g_loss
                       + lam      * conf_loss
                       + lam_sob  * sob_loss
                       + lam_perc * perc_loss
                       + lam_fft  * f_loss)
            total_g.backward()
            opt_gen.step()

            # ===== Refiner step =====
            refiner.zero_grad()
            with torch.no_grad():
                pg2, _, _, pr2 = generator(low_rgb)
            pred_rgb2 = refiner(pr2.detach(), pg2.detach())
            r_loss = ref_loss_fn(normal_rgb, pred_rgb2)
            r_loss.backward()
            opt_ref.step()

            total_conf += conf_loss.item()
            total_sob  += sob_loss.item()
            total_perc += perc_loss.item()
            total_fft  += f_loss.item()

            loop.set_postfix(
                Conf=f"{conf_loss.item():.3f}",
                Sob=f"{sob_loss.item():.3f}",
                Perc=f"{perc_loss.item():.3f}",
                FFT=f"{f_loss.item():.2f}"
            )

        n = len(train_dataloader)
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Conf:{total_conf/n:.4f} Sob:{total_sob/n:.4f} "
            f"Perc:{total_perc/n:.4f} FFT:{total_fft/n:.2f}"
        )

        torch.save(generator.state_dict(), os.path.join(save_dir, f'gen_epoch{epoch}.pth'))
        torch.save(refiner.state_dict(),   os.path.join(save_dir, f'ref_epoch{epoch}.pth'))

        # 50epoch마다 이전 체크포인트 정리 (디스크 절약, 최근 10개만 유지)
        if epoch >= 10:
            old_epoch = epoch - 10
            old_gen = os.path.join(save_dir, f'gen_epoch{old_epoch}.pth')
            old_ref = os.path.join(save_dir, f'ref_epoch{old_epoch}.pth')
            if old_epoch % 10 != 0:  # 10의 배수는 보존
                if os.path.exists(old_gen): os.remove(old_gen)
                if os.path.exists(old_ref): os.remove(old_ref)

    logger.info("FINAL FT done!")


if __name__ == "__main__":
    main()
