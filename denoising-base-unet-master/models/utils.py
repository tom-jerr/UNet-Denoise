import os
from torchvision.utils import save_image
from torchvision import transforms
import torch
import sys
sys.path.append('./denoising-base-unet-master')
def to_img(x):
    x = x.view(x.size(0), x.size(1), 32, 32)
    return x

def to_img_128(x):
    x = x.view(x.size(0), x.size(1), 128, 128)
    return x
def makeDirectories():
    if not os.path.exists('./denoising-base-unet-master/result_images'):
        os.mkdir('./denoising-base-unet-master/result_images')
    if not os.path.exists('./denoising-base-unet-master/saved_model'):
        os.mkdir('./denoising-base-unet-master/saved_model')
    if not os.path.exists('./denoising-base-unet-master/test_images'):
        os.mkdir('./denoising-base-unet-master/test_images')




def _save_image(ground_truth, noise, unet_output, epoch):
    save_image(ground_truth, '../result_images/ground_truth_{}.png'.format(epoch))
    save_image(noise, '../result_images/noise_{}.png'.format(epoch))
    save_image(unet_output, '../result_images/unet_output_{}.png'.format(epoch))

def _save_test_image(ground_truth, noise, unet_output, epoch):
    save_image(ground_truth, './denoising-base-unet-master/test_images/ground_truth_{}.png'.format(epoch))
    save_image(noise, './denoising-base-unet-master/test_images/noise_{}.png'.format(epoch))
    save_image(unet_output, './denoising-base-unet-master/test_images/unet_output_{}.png'.format(epoch))
def _to_img(img, noise_img, output):
    ground_truth = to_img(img.cpu().data)
    noise = to_img(noise_img.cpu().data)
    unet_output = to_img(output.cpu().data)
    return ground_truth, noise, unet_output

def _to_img_128(img, noise_img, output):
    ground_truth = to_img_128(img.data)
    noise = to_img_128(noise_img.data)
    unet_output = to_img_128(output.data)
    return ground_truth, noise, unet_output
def to_32_32_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
