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


def sobel_filter(img):
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=img.device)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=img.device)
    sobel_x = sobel_x.view(1,1,3,3).repeat(img.shape[1],1,1,1)
    sobel_y = sobel_y.view(1,1,3,3).repeat(img.shape[1],1,1,1)
    gx = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
    gy = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
    return torch.sqrt(gx**2 + gy**2 + 1e-6)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt",      type=str,   required=True)
    parser.add_argument("-lam",      type=float, default=10.0)
    parser.add_argument("-lam_sob",  type=float, default=0.5)
    parser.add_argument("-lam_perc", type=float, default=0.05)
    args = parser.parse_args()
    opt = parse(args.opt)
    opt["device"] = select_device()
    return opt, args.lam, args.lam_sob, args.lam_perc


def main():
    opt, lam, lam_sob, lam_perc = parse_config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"lam={lam}, lam_sob={lam_sob}, lam_perc={lam_perc}")

    train_dataloader, _ = get_data(opt)

    generator, refiner = get_model(opt['model'], device=opt['device'])

    # ft_sobel epoch99에서 이어서 시작
    ft_gen = r'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints\ft_lam10.0_sobel0.5\gen_epoch99.pth'
    ft_ref = r'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints\ft_lam10.0_sobel0.5\ref_epoch99.pth'

    def load_w(model, path):
        ckpt = torch.load(path, map_location=opt['device'])
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt)
        return model

    generator = load_w(generator, ft_gen)
    refiner   = load_w(refiner,   ft_ref)
    logger.info("ft_sobel epoch99 loaded!")

    opt_gen = optim.Adam(generator.parameters(), lr=1e-5)
    opt_ref = optim.Adam(refiner.parameters(),   lr=1e-5)

    gen_loss_fn = GenLoss()
    ref_loss_fn = RefineLoss()

    # LPIPS (perceptual loss)
    lpips_fn = lpips.LPIPS(net='vgg').to(opt['device'])
    for p in lpips_fn.parameters():
        p.requires_grad = False

    # LoFTR
    matcher = LoFTR(config=default_cfg)
    ckpt = torch.load(r'C:\Users\user\Desktop\17\repos\LOFTR\weights\indoor_ds_new.ckpt')
    matcher.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    matcher = matcher.eval().to(opt['device'])
    for p in matcher.parameters():
        p.requires_grad = False
    logger.info("LoFTR freeze done!")

    epochs = 50
    save_dir = fr'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints\ft_sobel_perc{lam_perc}'
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        generator.train()
        refiner.train()
        loop = tqdm(train_dataloader)
        total_conf = total_sob = total_perc = 0.0

        for low_rgb, normal_rgb, normal_gray in loop:
            low_rgb    = low_rgb.to(opt['device'])
            normal_rgb = normal_rgb.to(opt['device'])
            normal_gray = normal_gray.to(opt['device'])

            # Generator step
            generator.zero_grad()
            pred_gray, pred_tm, pred_al, pred_refined = generator(low_rgb)
            g_loss = gen_loss_fn(normal_gray, pred_tm.detach(), pred_al.detach(),
                                 pred_gray, pred_tm, pred_al)

            pred_rgb_g = refiner(pred_refined, pred_gray)

            # Sobel loss
            sob_loss = F.l1_loss(sobel_filter(pred_rgb_g), sobel_filter(normal_rgb))

            # Perceptual loss (LPIPS, [-1,1] 범위로 정규화)
            pred_norm = pred_rgb_g * 2 - 1
            gt_norm   = normal_rgb * 2 - 1
            perc_loss = lpips_fn(pred_norm, gt_norm).mean()

            # LoFTR confidence loss
            gray0 = (0.299*pred_rgb_g[:,0] + 0.587*pred_rgb_g[:,1] + 0.114*pred_rgb_g[:,2]).unsqueeze(1)
            gray1 = (0.299*normal_rgb[:,0]  + 0.587*normal_rgb[:,1]  + 0.114*normal_rgb[:,2]).unsqueeze(1)
            gray0 = F.interpolate(gray0, size=(480,640))
            gray1 = F.interpolate(gray1, size=(480,640))
            batch_loftr = {'image0': gray0, 'image1': gray1}
            matcher(batch_loftr)
            conf_loss = -batch_loftr['mconf'].mean()

            total_g = g_loss + lam*conf_loss + lam_sob*sob_loss + lam_perc*perc_loss
            total_g.backward()
            opt_gen.step()

            # Refiner step
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

            loop.set_postfix(
                Conf=f"{conf_loss.item():.3f}",
                Sob=f"{sob_loss.item():.3f}",
                Perc=f"{perc_loss.item():.3f}"
            )

        n = len(train_dataloader)
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Conf:{total_conf/n:.4f} Sob:{total_sob/n:.4f} Perc:{total_perc/n:.4f}"
        )

        torch.save(generator.state_dict(), os.path.join(save_dir, f'gen_epoch{epoch}.pth'))
        torch.save(refiner.state_dict(),   os.path.join(save_dir, f'ref_epoch{epoch}.pth'))

    logger.info("Sobel+Perceptual FT done!")


if __name__ == "__main__":
    main()
