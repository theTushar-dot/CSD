import torch
import numpy as np
import random
from skimage.segmentation import slic
from skimage.color import label2rgb
import cv2
from PIL import Image
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def convert_to_depth(depth_array):
    depth_image = Image.fromarray(depth_array)
    return depth_image

def preprocess_depth_arr(image_path):
    _, file_extension = os.path.splitext(image_path)

    if file_extension == '.npy':
        depth_array =  np.load(image_path)
    elif file_extension == '.png':
        image = Image.open(image_path)
        depth_array = np.array(image)[:, :, 0]  # Although depth image have 3 channel but all contains same information
    if depth_array.dtype != 'uint8':
        depth_array = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))
        depth_array = (depth_array * 255).astype(np.uint8)
    return depth_array

def convert_to_canny_image(depth_array):
    edges = cv2.Canny(depth_array, threshold1=10, threshold2=10)
    canny_image = Image.fromarray(edges)
    return canny_image

def depth_to_normal(depth_array):
    depth_array = Image.fromarray(depth_array)
    dzdx = np.gradient(depth_array, axis=1)
    dzdy = np.gradient(depth_array, axis=0)
    normal = np.dstack((-dzdx, -dzdy, np.ones_like(depth_array)))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    normal = (normal + 1) / 2 * 255
    normal_image = Image.fromarray(normal.astype(np.uint8))
    return normal_image


def segment_depth_image(depth_array):
    segments = slic(depth_array, n_segments=100, compactness=10)
    segmented_image = label2rgb(segments, image=depth_array, kind='avg')
    segmented_image = (segmented_image * 255).astype(np.uint8)
    segmented_image = Image.fromarray(segmented_image)
    return segmented_image