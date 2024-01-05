import os
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from data.data_RGB import get_test_data
from models.MISCFilterNet import MISCKernelNet as myNet
from skimage import img_as_ubyte
from tools.get_parameter_number import get_parameter_number
from tqdm import tqdm
from models.layers import *
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity
import cv2


parser = argparse.ArgumentParser(description='Image Deblurring')
parser.add_argument('--meta', default='./dataset/HIDE/HIDE_test_list.txt', type=str, help='Directory of validation images')
parser.add_argument('--input_dir', default='./dataset/HIDE', type=str, help='Directory of validation images')
parser.add_argument('--target_dir', default='./dataset/HIDE', type=str, help='Directory of validation images')
parser.add_argument('--output_dir', default='./results/HIDE', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='./checkpoints/GoPro.pth', type=str, help='Path to weights')
parser.add_argument('--get_psnr', default=True, type=bool, help='PSNR')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_result', action='store_true', help='save resulting image')
parser.add_argument('--win_size', default=256, type=int, help='window size, [GoPro, HIDE, RealBlur]=256, [DPDD]=512')
args = parser.parse_args()
result_dir = args.output_dir
win = args.win_size
get_psnr = args.get_psnr
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
model_restoration = myNet(inference=False)
# print number of model
get_parameter_number(model_restoration)
utils.load_checkpoint(model_restoration, args.weights)
# utils.load_checkpoint_compress_doconv(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# dataset = args.dataset
test_dataset = get_test_data(args.meta, args.input_dir, args.target_dir, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
psnr_val_rgb = []
psnr = 0

utils.mkdir(result_dir)

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


with torch.no_grad():
    psnr_list = []
    ssim_list = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        input_    = data_test[0].cuda()
        path_gt = data_test[1][0]
        filenames = data_test[2]
        _, _, Hx, Wx = input_.shape
        input_re, batch_list = window_partitionx(input_, win)
        restored,_ = model_restoration(input_re)
        restored = restored[0]
        restored = window_reversex(restored, win, Hx, Wx, batch_list)

        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = restored[batch]
            restored_img = img_as_ubyte(restored[batch])
            if get_psnr:
                rgb_gt = cv2.imread(os.path.join(args.target_dir, path_gt))
                rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB)
                psnr_val_rgb.append(psnr_loss(restored_img, rgb_gt))
            if args.save_result:
                utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)

if get_psnr:
    psnr = sum(psnr_val_rgb) / len(test_dataset)
    print("PSNR: %f" % psnr)
