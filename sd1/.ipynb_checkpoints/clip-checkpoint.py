import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchmetrics.multimodal.clip_score import CLIPScore

def load_prompts(txt_file):
    with open(txt_file, "r") as f:
        prompts = f.read().splitlines()
    return prompts

def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def load_image(image_folder, index):
    img_path = os.path.join(image_folder, f"{index:05}.png")
    image = Image.open(img_path).convert("RGB")
    image_tensor = ToTensor()(image).unsqueeze(0)  # 形状 (1, C, H, W)
    return image_tensor

def main(prompt_file="prompts.txt", image_folder="images", term=1000):
    # 加载prompts
    prompts = load_prompts(prompt_file)
    
    # 初始化CLIPScore对象
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_score_metric = CLIPScore(model_name_or_path="/root/autodl-tmp/openai/clip-vit-large-patch14").to(device)

    images_num = count_files(image_folder)
    
    # 处理每一个prompt
    for idx in range(term):
        # print(idx, '/', term)
        prompt = prompts[idx]
        k = 0
        while True:
            image_index = idx + term * k
            if image_index >= images_num:
                break
            image_tensor = load_image(image_folder, image_index).to(device)
            
            # 更新CLIPScore
            clip_score_metric.update(image_tensor.float(), prompt)
            k += 1

    # 计算并打印最终的CLIPScore
    final_clip_score = clip_score_metric.compute()
    print(image_folder)
    print(f"Final CLIP Score: {final_clip_score.item()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate CLIP Score for images and prompts.")
    parser.add_argument("--prompt_file", type=str, default="/root/autodl-tmp/stable-diffusion-main-1/coco.txt", help="Path to the prompts text file.")
    parser.add_argument("--image_folder", type=str, default="/root/autodl-tmp/stable-diffusion-main-1/outputs/txt2img-samples/imagenet/gtpr/sa_ca/ratio4/samples")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for the first set of images.")
    parser.add_argument("--term", type=int, default=1000, help="Term to increment the image index.")

    args = parser.parse_args()
    main(prompt_file=args.prompt_file, image_folder=args.image_folder, term=args.term)