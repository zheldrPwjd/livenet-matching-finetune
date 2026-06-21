# Low-Light Enhancement × Matching Fine-tuning (LIVENet)

화질(PSNR)은 유지하면서, 저조도 향상 모델 LIVENet을 매칭(LoFTR Inlier Ratio) 친화적으로
Fine-tuning하는 방법을 연구한 프로젝트입니다.

## 핵심 결과

| Model | PSNR | SSIM | LOL 20° | LOL 30° | Night Total |
|---|---|---|---|---|---|
| Original | 7.77 | 0.191 | 0.378 | 0.000 | 0.719 |
| Base (LIVENet) | 20.45 | 0.710 | 0.666 | 0.532 | 0.658 |
| FT-Sobel | 20.36 | 0.707 | 0.684 | 0.588 | 0.688 |
| **FT-Sobel-Perc** | 20.42 | 0.752 | 0.685 | 0.548 | **0.712** |
| FT-Final | 20.19 | 0.760 | 0.681 | 0.567 | 0.647 |

- **FT-Sobel**: 구조 변경 없이 Sobel edge loss만 추가 — 화질 거의 유지, LOL 전 구간 Inlier 개선
- **FT-Sobel-Perc**: Perceptual(LPIPS) loss 추가 — 야간 실환경 11쌍 실측에서 향상 모델 중 1위, SSIM도 함께 개선
- **FT-Final**: 구조 수정(hf_branch) + 4개 손실 결합 — LOL 통제실험 최고지만 실환경에서는 오히려 Base보다 낮음 (Occam's Razor)

최종 선택 모델: **FT-Sobel-Perc**

## 파일 구성

| 파일 | 설명 |
|---|---|
| `finetune_sobel.py` | Sobel edge loss 기반 fine-tuning |
| `finetune_sobel_perc.py` | Sobel + Perceptual(LPIPS) loss 기반 fine-tuning |
| `finetune_final.py` | 구조 수정(hf_branch) + 4개 손실 결합 fine-tuning |
| `finetune_seq.py` | 순차적(sequential) fine-tuning — ft_sobel_perc 체크포인트에서 hf_branch+FFT 추가 학습 |
| `eval_psnr_ssim.py` | PSNR / SSIM 측정 |
| `eval_final.py` | LOL eval15 각도별(10°~30°) Inlier Ratio 평가 |
| `eval_night_final.py` | 야간 실측 11쌍 Inlier Ratio 평가 |
| `blocks.py` | LIVENet 모델 구조 (hf_branch 추가 버전) |
| `blocks_backup.py` | LIVENet 모델 구조 (원본) |

## 사용 모델

- **LIVENet**: 저조도 이미지 향상 모델 (Generator + Refiner), 본 연구의 Fine-tuning 대상
- **LoFTR**: Transformer 기반 이미지 매칭 모델, 가중치 고정(frozen) 후 손실 계산·평가에만 사용

## 평가 지표

- **PSNR / SSIM**: 향상 이미지와 GT 사이의 화질 유사도
- **Inlier Ratio**: LoFTR 매칭점 중 RANSAC 기하 검증을 통과한 비율 (매칭 정확도)
