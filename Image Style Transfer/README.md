

# DL Mini Project

Fast Neural Style Transfer
This project implements a Fast Neural Style Transfer model to apply artistic styles to images. The model is trained using PyTorch and can transform any content image into the style of famous artists like Van Gogh or Picasso.
Project Structure

MiniProject.ipynb: The notebook for inference on local machine, containing the model architecture and code to process images.


Model Architecture
The model uses a transformer network with:

Encoder: Convolutional layers that extract features from the input image
Residual blocks: Maintain the content while transforming the style
Decoder: Upsampling layers that reconstruct the stylized image

Usage

Clone this repository
Place your images in an images directory
Run the notebook or script to apply style transfer:

python# Example usage
python style_transfer.py --input_dir images --output_dir stylized_images --model_path fast_style_transfer_model.pth
Results
The model transforms content images by applying the learned artistic style while preserving the original content.
References

Perceptual Losses for Real-Time Style Transfer and Super-Resolution by Johnson et al.
Based on PyTorch implementation of Fast Neural Style Transfer
