import argparse
import os
import cv2
import torch
import numpy as np
import math
from skimage.metrics import structural_similarity
from skimage import io


gt_path=r"" # 文件夹路径
result_path=r"" # 文件夹路径

# loss_fn_alex = lpips.LPIPS(net='alex')
# dists = DISTS()


def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)


def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False, data_range = 1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim


def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift



psnr_l = []
ssim_l = []
# lpips_l = []
# dists_l = []
subdir = sorted(os.listdir(gt_path))
for dir_name in subdir:
    subsubdir = sorted(os.listdir(os.path.join(gt_path,dir_name)))
    for img_name in subsubdir:
        GT_path = os.path.join(gt_path, dir_name, img_name)
        output_path = os.path.join(result_path, dir_name,img_name)
        print(GT_path)
        print(output_path)
        prd_img = io.imread(output_path)
        tar_img= io.imread(GT_path)

        tar_img = tar_img.astype(np.float32)/255.0
        prd_img = prd_img.astype(np.float32)/255.0
        
        prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)


        deblur_psnr = compute_psnr(tar_img, prd_img, cr1, data_range=1)
        deblur_ssim = compute_ssim(tar_img, prd_img, cr1)

        print(deblur_psnr,deblur_ssim)
        psnr_l.append(deblur_psnr)
        ssim_l.append(deblur_ssim)


avg_psnr = sum(psnr_l) / len(psnr_l)
avg_ssim = sum(ssim_l) / len(ssim_l)
print('*'*30)
print(avg_psnr,len(psnr_l))
print(avg_ssim,len(ssim_l))
