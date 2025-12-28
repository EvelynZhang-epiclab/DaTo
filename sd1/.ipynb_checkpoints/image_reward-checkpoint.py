import os
import torch
import ImageReward as RM
from tqdm import tqdm
from PIL import Image, ExifTags

def load_images_and_prompts(image_folder, prompt_source="exif", prompt_txt=None):
    """
    Load images and extract prompts based on the specified method.
    
    Args:
        image_folder (str): Path to the folder containing images.
        prompt_source (str): One of "exif", "prompt-number", or "prompt-prompt".
        prompt_txt (str): Path to the text file containing prompts (required for "prompt-number").
    
    Returns:
        images (list): List of image file paths.
        prompts (list): Corresponding list of prompts.
    """
    images = []
    prompts = []
    # List files with allowed extensions
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if prompt_source == "exif":
        # Use EXIF metadata for prompts
        for filename in image_files:
            img_path = os.path.join(image_folder, filename)
            try:
                img = Image.open(img_path)
                exif_data = img._getexif()
                if exif_data is not None:
                    exif = {}
                    for tag_id, value in exif_data.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif[tag] = value
                    prompt = exif.get('ImageDescription', None)
                    if prompt:
                        images.append(img_path)
                        prompts.append(prompt)
                    else:
                        print(f"No prompt found in image metadata: {filename}")
                else:
                    print(f"No EXIF metadata found in image: {filename}")
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
        return images, prompts

    elif prompt_source == "prompt-number":
        # Read prompts from a text file; pairing is done by the line number
        if prompt_txt is None:
            raise ValueError("For prompt-number pairing, --prompt_txt must be provided.")
        with open(prompt_txt, 'r', encoding='utf-8') as f:
            prompt_lines = [line.strip() for line in f if line.strip()]
        
        # Define a key to sort images numerically if possible
        def numeric_key(filename):
            base = os.path.splitext(filename)[0]
            try:
                return int(base)
            except ValueError:
                return base  # fallback to lexicographic order if not a number

        image_files_sorted = sorted(image_files, key=numeric_key)
        
        print(f"Number of prompts in txt: {len(prompt_lines)}")
        print(f"Number of images in folder: {len(image_files_sorted)}")
        if prompt_lines:
            print(f"Starting prompt: {prompt_lines[0]}")
            print(f"Ending prompt: {prompt_lines[-1]}")
        
        n = min(len(image_files_sorted), len(prompt_lines))
        if len(image_files_sorted) != len(prompt_lines):
            print(f"Warning: Number of images ({len(image_files_sorted)}) and prompts ({len(prompt_lines)}) do not match. Using first {n} items.")
        for i in range(n):
            img_path = os.path.join(image_folder, image_files_sorted[i])
            images.append(img_path)
            prompts.append(prompt_lines[i])
        return images, prompts

    elif prompt_source == "prompt-prompt":
        # Use the image filename as the prompt.
        # If EXIF metadata is available and its prompt is longer (i.e. not truncated), use that.
        for filename in image_files:
            img_path = os.path.join(image_folder, filename)
            base_prompt = os.path.splitext(filename)[0]
            full_prompt = None
            try:
                img = Image.open(img_path)
                exif_data = img._getexif()
                if exif_data is not None:
                    exif = {}
                    for tag_id, value in exif_data.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif[tag] = value
                    exif_prompt = exif.get('ImageDescription', None)
                    # If EXIF prompt exists and appears to be a longer version of the filename prompt, use it
                    if exif_prompt and len(base_prompt) < len(exif_prompt) and exif_prompt.startswith(base_prompt):
                        full_prompt = exif_prompt
                if full_prompt is None:
                    full_prompt = base_prompt
                images.append(img_path)
                prompts.append(full_prompt)
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
        return images, prompts

    else:
        raise ValueError("Unsupported prompt source. Choose from 'exif', 'prompt-number', or 'prompt-prompt'.")


def process_batch(images_batch, prompts_batch, model, device):
    """
    Process a batch of images and prompts to compute scores.
    
    Args:
        images_batch (list): List of image file paths in the batch.
        prompts_batch (list): Corresponding list of prompts.
        model: ImageReward model instance.
        device: PyTorch device.
    
    Returns:
        scores (list): List of computed scores for the batch.
    """
    scores = []
    for prompt, img_path in zip(prompts_batch, images_batch):
        with torch.no_grad():
            score = model.score(prompt, img_path)
            scores.append(score)
    return scores


def main(image_folder, batch_size, model_name, prompt_source, prompt_txt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available!")
    
    # Load the model
    model = RM.load(model_name).to(device)
    print(f"Loaded model '{model_name}' on {device}")
    
    # Load images and prompts according to the selected prompt source
    images, prompts = load_images_and_prompts(image_folder, prompt_source, prompt_txt)
    
    if not images:
        print("No images with prompts found in the specified folder.")
        return
    
    total_batches = (len(images) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Processing batches")
    
    all_scores = []
    for i in range(0, len(images), batch_size):
        images_batch = images[i:i + batch_size]
        prompts_batch = prompts[i:i + batch_size]
        scores = process_batch(images_batch, prompts_batch, model, device)
        all_scores.extend(scores)
        progress_bar.update(1)
    progress_bar.close()
    
    average_score = sum(all_scores) / len(all_scores)
    print(f"Final average score: {average_score}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Single-GPU ImageReward scoring script with multiple prompt pairing options.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing.")
    parser.add_argument("--model_name", type=str, default="ImageReward-v1.0", help="ImageReward model name.")
    parser.add_argument("--prompt_source", type=str, choices=["exif", "prompt-number", "prompt-prompt"], default="exif",
                        help="Method to obtain prompts. 'exif' reads from image metadata, 'prompt-number' uses a txt file with line-number pairing, and 'prompt-prompt' uses the filename as prompt (with matching if available).")
    parser.add_argument("--prompt_txt", type=str, default=None,
                        help="Path to the text file containing prompts (required for 'prompt-number' pairing).")
    
    args = parser.parse_args()
    main(image_folder=args.image_folder,
         batch_size=args.batch_size,
         model_name=args.model_name,
         prompt_source=args.prompt_source,
         prompt_txt=args.prompt_txt)
