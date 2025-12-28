import os
import torch
import clip
from PIL import Image
from urllib.request import urlretrieve
from tqdm import tqdm  # for progress bar
import numpy as np

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocessor
model, preprocess = clip.load("ViT-L/14", device=device)

# Function to download and load LAION aesthetic score linear model
def get_aesthetic_model(clip_model="vit_l_14"):
    home = os.path.expanduser("~")
    cache_folder = os.path.join(home, ".cache", "aesthetic_predictor")
    os.makedirs(cache_folder, exist_ok=True)
    model_path = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")
    if not os.path.exists(model_path):
        url = f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_{clip_model}_linear.pth?raw=true"
        urlretrieve(url, model_path)
    # Load appropriate linear layer based on CLIP model type
    if clip_model == "vit_l_14":
        linear_model = torch.nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        linear_model = torch.nn.Linear(512, 1)
    else:
        raise ValueError("Unsupported CLIP model")
    # Load pretrained weights
    state_dict = torch.load(model_path)
    linear_model.load_state_dict(state_dict)
    linear_model = linear_model.to(device)
    linear_model.eval()
    return linear_model

# Load aesthetic score model
aesthetic_model = get_aesthetic_model("vit_l_14")

# Function to preprocess and batch images
def preprocess_images_batch(image_paths):
    images = []
    for image_path in image_paths:
        image = preprocess(Image.open(image_path)).unsqueeze(0)
        images.append(image)
    return torch.cat(images).to(device)

# Function to compute aesthetic scores for a batch of images
def evaluate_images_batch(images):
    with torch.no_grad():
        image_features = model.encode_image(images).to(torch.float32)  # Ensure features are float32
        scores = aesthetic_model(image_features).squeeze().tolist()
    return scores


# Function to evaluate all images in a folder with batching
def evaluate_folder_aesthetic(folder_path, batch_size=8):
    scores = []
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Evaluating images in batches"):
        batch_paths = image_paths[i:i + batch_size]
        try:
            images = preprocess_images_batch(batch_paths)
            batch_scores = evaluate_images_batch(images)
            scores.extend(batch_scores)
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")

    mean_score = np.mean(scores) if scores else 0
    return mean_score, scores
    
x_values= [0.3]# [0.3,0.4,0.5,0.6,0.7]
for x in x_values:
    folder_path = f"/dockerdata/yuuweizhang/projects/sd2/outputs/tome/coco30k/ratio_{x}/samples"  # Replace with your folder path
    batch_size = 200  # Set the desired batch size
    
    # Evaluate aesthetic scores and print the results
    print("Starting evaluation of images in the folder...")
    mean_score, all_scores = evaluate_folder_aesthetic(folder_path, batch_size=batch_size)
    print(f"Average aesthetic score{folder_path}: {mean_score:.2f}")

