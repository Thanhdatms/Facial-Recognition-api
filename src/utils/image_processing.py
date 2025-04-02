from PIL import Image
import torchvision.transforms as transforms
import torch

# Define transform for inference
inference_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def encode_image(file_path, model):
    """Encode image to a 128-dimension embedding using custom model."""
    try:
        img = Image.open(file_path).convert("RGB")
        img_tensor = inference_transform(img).unsqueeze(0).to(model.device)
        with torch.no_grad():
            embedding = model.embedding(img_tensor).cpu().numpy().flatten()
        return embedding
    except Exception as e:
        print(f"Error encoding {file_path}: {e}")
        return None