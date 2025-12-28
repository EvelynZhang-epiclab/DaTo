import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore
#定义图像的预处理，仅调整尺寸，不进行标准化
transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x:(x * 255).byte()) 
])# 将浮点数转换为 uint8 类型

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__ (self,root_dir, transform=None):
        self.root_dir = root_dir
        #过滤出所有png文件的路径
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.png')]
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 打开图像并转换为RGB
        image = Image.open(img_path).convert("RGB")
        # 应用预处理变换
        if self.transform:
            image = self.transform(image)
        return image
        
x_values= [0.3,0.4,0.5,0.6,0.7]
# 指定文件路径
file_path = '/dockerdata/yuuweizhang/projects/sd2/outputs/sito/coco30k/IS.txt'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
for x in x_values:
    #设置数据路径
    data_path =f'/dockerdata/yuuweizhang/projects/sd2/outputs/sito/coco30k/ratio_{x}/samples'
    # 加载图片数据集
    dataset = ImageDataset(root_dir=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 初始化 Inceptionscore 模块
    inception = InceptionScore().to("cuda:0")
    
    # 遍历数据集并更新 Inception Score
    
    for images in dataloader:
        images = images.to("cuda:0")
        inception.update(images)
    
    #计算最终的 Inception Score
    score = inception.compute()
    print(f'Ratio {x} ,Inception score: {score}')
    with open(file_path, 'a') as file:
        file.write(f'ratio {x}: {score}\n')  # 写入内容并换行
