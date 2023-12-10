import sys, os
sys.path.append('./denoising-base-unet-master')
# print(sys.path)
from models.DenseBlock import DenseBlock
from models.DownBlock import DownBlock
from models.UpBlock import UpBlock
from models.TransitionBlock import TransitionBlock
from models.BaseUNet import BaseUNet
from PIL import Image
from torchvision import transforms
import models.utils as utils
import torch 
from torch.autograd import Variable

utils.makeDirectories()
batch_size = 128

model = BaseUNet(3, 3)
# model.cuda()
model.eval()
model.load_state_dict(torch.load('./denoising-base-unet-master/saved_model/cifar10_base_unet.pth', map_location='cpu'))

# data = CIFAR10('./data', download=True, transform=utils.to_32_32_transform(),train=False)
# dataset = DataLoader(data, batch_size=batch_size, shuffle=True)
#
#
#
# avg_psnr = 0
# for img, _ in dataset:

transfrom_valid = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
img_path_vec = ['./pic/docker_original.png', './pic/img.png']
print("input image")
total_psnr = 0
# for i,img_path in enumerate(img_path_vec):
#     img = Image.open(img_path)
#     img = transfrom_valid(img).unsqueeze(0)
#     # img = img.cuda()
#     noise_img = Variable(
#         img+img.data.new(img.size()).normal_(0.0, 0.1))
#     output = model(noise_img)
#     MSE_loss = torch.nn.MSELoss()(output, img)
#     psnr = 10 * torch.log10(1 / MSE_loss.data)
#     noise_psnr = 10 * torch.log10(1 / (torch.nn.MSELoss()(noise_img, img)).data)
#     total_psnr += psnr
#     noise_loss = torch.nn.MSELoss()(noise_img, img)
#     print('Image:{} MSE_loss:{:.4f}'.format(i, MSE_loss.data))
#     print('Image:{} noise_loss:{:.4f}'.format(i, noise_loss.data))
#     print("Image:{} img_PSNR: {:.4f} dB".format(i, psnr))
#     print("Image:{} noise_PSNR: {:.4f} dB".format(i, noise_psnr))

#     ground_truth,noise,unet_output= utils._to_img_128(img, noise_img, output)
#     utils._save_test_image(ground_truth, noise, unet_output, epoch=i)
# print("test finished")
# print("average psnr: {:.4f} dB".format(total_psnr/len(img_path_vec)))
import time
img_path = './pic/docker_original.png'
start = time.time()
for _ in range(50):

    img = Image.open(img_path)
    img = transfrom_valid(img).unsqueeze(0)
    # img = img.cuda()


    noise_img = Variable(
        img+img.data.new(img.size()).normal_(0.0, 0.1))
    output = model(noise_img)
end = time.time()
print('time:{} ms'.format((end-start)*1000))



