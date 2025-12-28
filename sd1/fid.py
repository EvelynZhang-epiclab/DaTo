# python -m pytorch_fid /root/autodl-tmp/imagenet50k.npz /root/autodl-tmp/stable-diffusion-main/outputs/txt2img-samples/imagenet/gtpr/ps2/ratio7/samples
import os
import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import models, transforms
from PIL import Image
from scipy.linalg import sqrtm
# rm -rf /root/autodl-tmp/stable-diffusion-main/outputs/txt2img-samples/gtpr/v7/ratio5/samples/.ipynb_checkpoints
# 定义图像目录路径
real_images_folder = '/root/autodl-tmp/selected_images'
generated_images_folder = '/root/autodl-tmp/stable-diffusion-main/outputs/txt2img-samples/gtpr/v7/ratio5/samples'

# 确保图像目录存在
assert os.path.exists(real_images_folder), f"Real images folder does not exist: {real_images_folder}"
assert os.path.exists(generated_images_folder), f"Generated images folder does not exist: {generated_images_folder}"

# 图像预处
transform = transforms.Compose([
    transforms.Resize((512, 512)), # 512, 512
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载InceptionV3模型
class InceptionV3(torch.nn.Module):
    def __init__(self, output_blocks, resize_input=True, normalize_input=True, requires_grad=False):
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        self.blocks = torch.nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(torch.nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                torch.nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(torch.nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e
            ]
            self.blocks.append(torch.nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(torch.nn.Sequential(*block3))

        # Block 4: final avgpool to logits
        if self.last_needed_block >= 4:
            block4 = [
                inception.fc
            ]
            self.blocks.append(torch.nn.Sequential(*block4))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        out = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                out.append(x)

            if idx == self.last_needed_block:
                break

        return out

block_idx = InceptionV3([3], normalize_input=False)
model = InceptionV3([3]).cuda()

def get_activations(files, model, batch_size=32, dims=2048, cuda=True):
    model.eval()
    d0 = dims
    d1 = len(files)
    pred_arr = np.empty((d1, d0))

    for i in range(0, d1, batch_size):
        start = i
        end = min(i + batch_size, d1)
        
        images = [transform(Image.open(str(f)).convert('RGB')) for f in files[start:end]]
        images = torch.stack(images)

        if cuda:
            images = images.cuda()

        with torch.no_grad():
            pred = model(images)[0]

        # If model output is not scalar, it should be array-like with the second dimension being the activation layer.
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    return pred_arr

def calculate_activation_statistics(files, model, batch_size=32, dims=2048, cuda=True):
    act = get_activations(files, model, batch_size, dims, cuda)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component {}".format(np.max(np.abs(covmean.imag))))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

# 获取图像文件列表
real_images = [os.path.join(real_images_folder, img) for img in os.listdir(real_images_folder)]
generated_images = [os.path.join(generated_images_folder, img) for img in os.listdir(generated_images_folder)]

# 计算真实图像和生成图像的统计数据
m1, s1 = calculate_activation_statistics(real_images, model, batch_size=32, dims=2048, cuda=True)
m2, s2 = calculate_activation_statistics(generated_images, model, batch_size=32, dims=2048, cuda=True)

# 计算FID
fid_value = calculate_frechet_distance(m1, s1, m2, s2)
print(f"FID: {fid_value}")

'''
python -m pytorch_fid /root/autodl-tmp/selected_images /root/autodl-tmp/tomesd_result/merge5/samples
'''