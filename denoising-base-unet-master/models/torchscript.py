import torch
import torchvision
from torchvision import transforms
import utils
from PIL import Image
from torch.autograd import Variable
from BaseUNet import BaseUNet



utils.makeDirectories()
batch_size = 128

model = BaseUNet(3, 3)
# model.cuda()
model.eval()
model.load_state_dict(torch.load('../saved_model/cifar10_base_unet.pth', map_location='cpu'))


# An example input you would normally provide to your model's forward() method.

transfrom_valid = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
img_path = '../img.png'
print("input image")
img = Image.open(img_path)
img = transfrom_valid(img).unsqueeze(0)
# img = img.cuda()
noise_img = Variable(
    img+img.data.new(img.size()).normal_(0.0, 0.1))

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.

traced_script_module = torch.jit.trace(model, noise_img)
traced_script_module.save('../saved_model/libtorch-model-gpu.pt')

# model.cpu()
# example = torch.rand(1, 3, 224, 224)
# traced_script_module = torch.jit.trace(model, example)
# traced_script_module.save("model-cpu.pt")