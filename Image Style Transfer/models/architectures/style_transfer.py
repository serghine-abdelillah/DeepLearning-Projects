#!/usr/bin/env python3
"""
Fast Neural Style Transfer Script
Apply artistic styles to images using a pre-trained model.
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import glob
import argparse
from tqdm import tqdm


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.layer = nn.Sequential(
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, 3, 1),
            ConvLayer(channels, channels, 3, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.encoder = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2),
        )
        self.residuals = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvLayer(128, 64, 3, 1),
            nn.Upsample(scale_factor=2),
            ConvLayer(64, 32, 3, 1),
            ConvLayer(32, 3, 9, 1),
            nn.Tanh(),  # [-1, 1] output
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        x = self.decoder(x)
        return x


def process_directory(input_dir, output_dir, model_path, image_size=256):
    """Process all images in a directory"""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print("Loading model...")
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    transformer.eval()
    print("Model loaded successfully!")
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # Normalize to [-1, 1]
    ])
    
    # Find all image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Get filename without directory
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"styled_{filename}")
            
            # Load and preprocess the image
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Perform style transfer
            with torch.no_grad():
                output_tensor = transformer(input_tensor)
            
            # Save the output image
            output = output_tensor[0].cpu()
            output = output * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
            output = output.clamp(0, 1)
            output_image = transforms.ToPILImage()(output)
            output_image.save(output_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Processing complete! Stylized images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply style transfer to a directory of images')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save stylized images')
    parser.add_argument('--model_path', type=str, default='fast_style_transfer_model.pth', help='Path to the trained model')
    parser.add_argument('--image_size', type=int, default=256, help='Size to resize images to')
    args = parser.parse_args()
    
    # Call the function with the parsed arguments
    process_directory(args.input_dir, args.output_dir, args.model_path, args.image_size)