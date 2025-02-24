import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/mead_90.png', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='output',
                        help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true',
                        help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true',
                        help='Present the result w.r.t. the original image/video size')

    args = parser.parse_args()

    input_image_size = 224

