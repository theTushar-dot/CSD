from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler, DDPMScheduler, HeunDiscreteScheduler, UniPCMultistepScheduler, EulerAncestralDiscreteScheduler
from PIL import Image
import PIL
from transformers import pipeline
import numpy as np
import argparse
import torch
from diffusers.utils import load_image
import cv2
import time
import tomesd
from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim
from skimage.color import label2rgb
from utils import set_seed, convert_to_canny_image, depth_to_normal, segment_depth_image, preprocess_depth_arr, convert_to_depth


def parse_args():
    parser = argparse.ArgumentParser(description='ControlledStableDiffusion(CSD)')

    parser.add_argument('--seed', type=int, default=12345, help='Sets the seed for random number generation to ensure reproducibility of results.')
    parser.add_argument('--use_cuda', action='store_true', help='If specified, the program will utilize GPU acceleration for computations')
    parser.add_argument('--prompt', type=str, default=None, help='The text prompt to guide the image generation process.')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Specifies the number of inference steps to be used in the generation process')
    parser.add_argument('--height', type=int, default=None, help='Sets the height of the generated image in pixels')
    parser.add_argument('--width', type=int, default=None, help=' Sets the width of the generated image in pixels')
    parser.add_argument('--beta_start', type=float, default=0.00085, help='The starting value of the beta parameter for the noise schedule in the diffusion process.')
    parser.add_argument('--beta_end', type=float, default=0.012, help='The ending value of the beta parameter for the noise schedule in the diffusion process')
    parser.add_argument('--control_guidance_start', type=float, default=0.0, help='The starting value for control guidance strength')
    parser.add_argument('--control_guidance_end', type=float, default=1.0, help='The ending value for control guidance strength')
    parser.add_argument('--controlnet_con_scale', nargs='+', type=float, default=1.0, help='Scale factors for control net contribution, allowing multiple values for multiple ControlNet inputs')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help=' The scale of guidance used in the generation process to steer the output towards the prompt')
    parser.add_argument('--scheduler', type=str, default="LMSD",  choices=["DDIM", "PNDM", "LMSD", "DDPM", "Heun"], help='Specifies the type of scheduler to use for the generation process')
    parser.add_argument('--input_img_pth', type=str, default=None, help='The file path of the input image to be used in the generation process')
    parser.add_argument('--generated_img_pth', type=str, default=None, help='The file path where the generated image will be saved')
    parser.add_argument('--control_with_depth', action='store_true', help='If specified, the generation process will be controlled using a depth image')
    parser.add_argument('--control_with_canny', action='store_true', help='If specified, the generation process will be controlled using Canny edge detection')
    parser.add_argument('--control_with_normals', action='store_true', help='If specified, the generation process will be controlled using surface normal information')
    parser.add_argument('--control_with_segment', action='store_true', help=' If specified, the generation process will be controlled using segmentation information')
    parser.add_argument('--use_f16', action='store_true', help='If specified, the program will use 16-bit floating point precision for faster computation')
    parser.add_argument('--use_resizing', action='store_true', help='If specified, the ControlNet input images will be resized to 512x512')
    parser.add_argument('--use_xFormers', action='store_true', help='If specified, model will use xFormers acceleration and token merging')

    args = parser.parse_args()
    return args
    

def generate_an_image(args):

    if args.use_f16:
        dtype_to_be_used =  torch.float16
    else:
        dtype_to_be_used =  torch.float32

    if args.use_cuda:
        torch_device = torch.device("cuda")


    all_control_nets = []
    if args.control_with_depth:
        print('loading model for depth...')
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype = dtype_to_be_used)
        controlnet.to(torch_device)
        all_control_nets.append(controlnet)
    if args.control_with_canny:
        print('loading model for canny...')
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype = dtype_to_be_used)
        controlnet.to(torch_device)
        all_control_nets.append(controlnet)
    if args.control_with_normals:
        print('loading model for normals...')
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype = dtype_to_be_used)
        controlnet.to(torch_device)
        all_control_nets.append(controlnet)
    if args.control_with_segment:
        print('loading model for segment...')
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype = dtype_to_be_used)
        controlnet.to(torch_device)
        all_control_nets.append(controlnet)

    if len(all_control_nets) > 1:
        print('using multiple nets')
        controlnet = all_control_nets
    elif len(all_control_nets) == 1:
        controlnet = all_control_nets[0]
    else:
        print('Error')

    pipeline = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet,  torch_dtype=dtype_to_be_used) 


    if args.scheduler == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif args.scheduler == "PNDM":
        pipeline.scheduler = PNDMScheduler.from_config(pipeline.scheduler.config)
    elif args.scheduler == "LMSD":
        pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif args.scheduler == "DDPM":
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    elif args.scheduler == "Heun":
        pipeline.scheduler = HeunDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        print('Please select a valid scheduler')

    if args.use_xFormers:
        pipeline.enable_xformers_memory_efficient_attention()

    pipeline.enable_model_cpu_offload()

    depth_image_path = args.input_img_pth
    depth_arr = preprocess_depth_arr(depth_image_path)

    all_img_for_controlnet = []
    if args.control_with_depth:
        print('loading image for depth...')
        image_for_controlnet = convert_to_depth(depth_arr)
        all_img_for_controlnet.append(image_for_controlnet)

    if args.control_with_canny:
        print('loading image for canny...')
        image_for_controlnet = convert_to_canny_image(depth_arr)
        all_img_for_controlnet.append(image_for_controlnet)

    if args.control_with_normals:
        print('loading image for normals...')
        image_for_controlnet = depth_to_normal(depth_arr)
        all_img_for_controlnet.append(image_for_controlnet)

    if args.control_with_segment:
        print('loading image for segment...')
        image_for_controlnet = segment_depth_image(depth_arr)
        all_img_for_controlnet.append(image_for_controlnet)


    if len(all_img_for_controlnet) == 1:
        image_for_controlnet = all_img_for_controlnet[0]
    else:
        image_for_controlnet = all_img_for_controlnet


    if isinstance(image_for_controlnet, list):
        ori_height = image_for_controlnet[0].size[1]
        ori_width = image_for_controlnet[0].size[0]
    else:
        ori_height = image_for_controlnet.size[1]
        ori_width = image_for_controlnet.size[0]


    is_img_good_to_go = 1000 > ori_height > 500 and 1000 > ori_width > 500

    if not is_img_good_to_go and args.use_resizing:
        print('Resizing Image...')
        if isinstance(image_for_controlnet, list):
            image_for_controlnet = [i_img.resize((512, 512), resample=PIL.Image.Resampling.LANCZOS) for i_img in image_for_controlnet]
        else:
            image_for_controlnet = image_for_controlnet.resize((512, 512), resample=PIL.Image.Resampling.LANCZOS)
    else:
        print('generating image with lower and extreme sizes')

    prompt =  args.prompt 
    generator = torch.manual_seed(args.seed)

    start_time = time.time()
    generated_images = pipeline(prompt, height = args.height, width = args.width,  image = image_for_controlnet, num_inference_steps=args.num_inference_steps,  generator=generator, controlnet_conditioning_scale = args.controlnet_con_scale, guidance_scale = args.guidance_scale, safety_checker=None)
    end_time = time.time()
    duration = end_time - start_time

    print(f"###Time taken for the task: {duration} seconds")   

    generated_image = generated_images.images[0]
    if not is_img_good_to_go and args.use_resizing:
        generated_image = generated_image.resize((ori_height, ori_width), resample=PIL.Image.Resampling.LANCZOS)
    generated_image.save(args.generated_img_pth)


    

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    generated_img = generate_an_image(args)
