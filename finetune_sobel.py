import argparse
import logging
import sys
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.append(r'C:\Users\user\Desktop\17\repos\LOFTR')
from src.loftr import LoFTR, default_cfg

from models import get_model, GenLoss, RefineLoss, get_refined_image
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
    parser.add_argument("-opt", type=str, required=True)
    parser.add_argument("-lam", type=float, default=10.0)
    parser.add_argument("-lam_sob", type=float, default=0.5)
    args = parser.parse_args()
    opt = parse(args.opt)
    opt["device"] = select_device()
    return opt, args.lam, args.lam_sob


def main():
    opt, lam, lam_sob = parse_config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Lambda_conf: {lam}, Lambda_sobel: {lam_sob}")

    train_dataloader, _ = get_data(opt)

    generator, refiner = get_model(opt["model"], device=opt["device"])

    ft_gen = r'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints\ft_lam_10.0\gen_epoch99.pth'
    ft_ref = r'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints\ft_lam_10.0\ref_epoch99.pth'
    generator.load_state_dict(torch.load(ft_gen, map_location=opt["device"]))
    refiner.load_state_dict(torch.load(ft_ref, map_location=opt["device"]))
    logger.info("ft_lam_10 epoch99 loaded!")

    opt_gen = optim.Adam(generator.parameters(), lr=1e-5)
    opt_ref = optim.Adam(refiner.parameters(), lr=1e-5)

    gen_loss_fn = GenLoss()
    ref_loss_fn = RefineLoss()

    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(
        torch.load(r'C:\Users\user\Desktop\17\repos\LOFTR\weights\indoor_ds_new.ckpt')['state_dict']
    )
    matcher = matcher.eval().to(opt["device"])
    for param in matcher.parameters():
        param.requires_grad = False
    logger.info("LoFTR freeze done!")

    epochs = 100

    for epoch in range(epochs):
        generator.train()
        refiner.train()
        loop = tqdm(train_dataloader)
        total_conf_loss = 0.0
        total_sob_loss = 0.0

        for batch_idx, (low_image_rgb, normal_image_rgb, normal_image_gray) in enumerate(loop):
            low_image_rgb     = low_image_rgb.to(opt["device"])
            normal_image_rgb  = normal_image_rgb.to(opt["device"])
            normal_image_gray = normal_image_gray.to(opt["device"])

            orig_tm, orig_al, orig_refined = get_refined_image(low_image_rgb, normal_image_rgb, opt)

            # ===== Generator step =====
            generator.zero_grad()
            pred_gray, pred_tm, pred_al, pred_refined = generator(low_image_rgb)
            pred_rgb_detached = refiner(pred_refined.detach(), pred_gray.detach())

            g_loss = gen_loss_fn(normal_image_gray, orig_tm, orig_al, pred_gray, pred_tm, pred_al)

            # conf loss for generator
            pred_rgb_g = refiner(pred_refined, pred_gray)
            gray0 = (0.299*pred_rgb_g[:,0] + 0.587*pred_rgb_g[:,1] + 0.114*pred_rgb_g[:,2]).unsqueeze(1)
            gray1 = (0.299*normal_image_rgb[:,0] + 0.587*normal_image_rgb[:,1] + 0.114*normal_image_rgb[:,2]).unsqueeze(1)
            gray0 = F.interpolate(gray0, size=(480,640))
            gray1 = F.interpolate(gray1, size=(480,640))
            batch_loftr = {'image0': gray0, 'image1': gray1}
            matcher(batch_loftr)
            conf_loss = -batch_loftr['mconf'].mean()
            sob_loss = F.l1_loss(sobel_filter(pred_rgb_g), sobel_filter(normal_image_rgb))

            total_g = g_loss + lam * conf_loss + lam_sob * sob_loss
            total_g.backward()
            opt_gen.step()

            # ===== Refiner step =====
            refiner.zero_grad()
            with torch.no_grad():
                pred_gray2, _, _, pred_refined2 = generator(low_image_rgb)
            pred_rgb2 = refiner(pred_refined2.detach(), pred_gray2.detach())
            r_loss = ref_loss_fn(normal_image_rgb, pred_rgb2)
            r_loss.backward()
            opt_ref.step()

            total_conf_loss += conf_loss.item()
            total_sob_loss += sob_loss.item()
            loop.set_postfix(
                GenLoss=g_loss.item(),
                ConfLoss=conf_loss.item(),
                SobelLoss=sob_loss.item()
            )

        avg_conf = total_conf_loss / len(train_dataloader)
        avg_sob  = total_sob_loss  / len(train_dataloader)
        logger.info(f"Epoch {epoch}/{epochs} | ConfLoss: {avg_conf:.4f} | SobelLoss: {avg_sob:.4f}")

        save_dir = fr'C:\Users\user\Desktop\17\repos\LIVENET\checkpoints\ft_lam{lam}_sobel{lam_sob}'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(generator.state_dict(), os.path.join(save_dir, f'gen_epoch{epoch}.pth'))
        torch.save(refiner.state_dict(), os.path.join(save_dir, f'ref_epoch{epoch}.pth'))

    logger.info("Sobel FT done!")


if __name__ == "__main__":
    main()
