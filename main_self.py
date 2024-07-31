from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, ControlNetModel, LMSDiscreteScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler, LMSDiscreteScheduler, HeunDiscreteScheduler, ScoreSdeVeScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.controlnet  import MultiControlNetModel
from diffusers.utils.torch_utils import randn_tensor
# from diffusers.pipelines.controlnet.
import argparse
import torch
from utils import set_seed, preprocess_depth_arr, convert_to_canny_image, depth_to_normal, segment_depth_image, convert_to_depth
from tqdm.auto import tqdm
from skimage.metrics import structural_similarity as ssim
import cv2



def parse_args():
    parser = argparse.ArgumentParser(description='ControlledStableDiffusion(CSD)')

    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--use_cuda', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--prompt', type=str, default=None, help='')
    parser.add_argument('--num_inference_steps', type=int, default=20, help='')
    parser.add_argument('--beta_start', type=float, default=0.00085, help='')
    parser.add_argument('--beta_end', type=float, default=0.012, help='')
    parser.add_argument('--control_guidance_start', type=float, default=0.0, help='')
    parser.add_argument('--control_guidance_end', type=float, default=1.0, help='')
    parser.add_argument('--controlnet_con_scale', nargs='+', type=float, default=1.0, help='')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='')
    parser.add_argument('--scheduler', type=str, default="DDIM",  choices=['DDIM'], help='')
    parser.add_argument('--input_img_pth', type=str, default=None, help='')
    parser.add_argument('--generated_img_pth', type=str, default=None, help='')
    parser.add_argument('--control_with_depth', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--control_with_canny', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--control_with_normals', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--control_with_segment', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--use_f16', action='store_true', help='Use learning rate scheduler')

    args = parser.parse_args()
    return args


def hey_u_generate_image(args):

    if args.use_f16:
        dtype_to_be_used =  torch.float16
    else:
        dtype_to_be_used =  torch.float32

    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype = dtype_to_be_used)  # TODO check Tiny as well 
    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype = dtype_to_be_used)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype = dtype_to_be_used)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype = dtype_to_be_used)


    # print(tokenizer.dtype)

    if args.use_cuda:
        torch_device = torch.device("cuda")
        vae.to(torch_device)
        text_encoder.to(torch_device)
        unet.to(torch_device) 

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
        controlnet = MultiControlNetModel(all_control_nets)
    elif len(all_control_nets) == 1:
        controlnet = all_control_nets[0]
    else:
        print('Error') #TODO throw an error

    
    if args.scheduler == 'EDS':
        scheduler = EulerDiscreteScheduler(beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000)
    else:
        # scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)  # TODO see config values for the scheduler
        print('using UNI...')
        scheduler = LMSDiscreteScheduler(beta_start=args.beta_start, beta_end=args.beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000)


    prompt = args.prompt       
    generator = torch.manual_seed(12345) # TODO check if   set_seed already taken care of this
    batch_size = 1

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    control_image_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=False
    )
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True)
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
        image_for_preprocess = all_img_for_controlnet[0]
    else:
        image_for_preprocess = all_img_for_controlnet

    image = control_image_processor.preprocess(image_for_preprocess).to(dtype= dtype_to_be_used)
    repeat_by = batch_size
    image = image.repeat_interleave(repeat_by, dim=0)
    image = image.to(device=torch_device, dtype=dtype_to_be_used)
    image = torch.cat([image] * 2)
    height, width = image.shape[-2:]

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    text_embeddings = text_embeddings.detach()

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length,truncation=True, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    # print(text_embeddings)
    # time()


    shape_of_latent = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
    latents = torch.randn(shape_of_latent, generator=generator, device='cpu', dtype=dtype_to_be_used).to(torch_device)
    scheduler.set_timesteps(args.num_inference_steps)
    latents = latents * scheduler.init_noise_sigma



    # controlnet_conditioning_scale = [1.0, 0.75]

    controlnet_conditioning_scale = args.controlnet_con_scale
    print(controlnet_conditioning_scale)

    mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
    control_guidance_start, control_guidance_end = (
        mult * [args.control_guidance_start],
        mult * [args.control_guidance_end],
    )

    controlnet_keep = []
    for i in range(len(scheduler.timesteps)):
        keeps = [
            1.0 - float(i / len(scheduler.timesteps) < s or (i + 1) / len(scheduler.timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)


    scheduler.set_timesteps(args.num_inference_steps)
    counter = 0

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        control_model_input = latent_model_input
        controlnet_prompt_embeds = text_embeddings

        if isinstance(controlnet_keep[counter], list):
            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[counter])]
        else:
            controlnet_cond_scale = controlnet_conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            cond_scale = controlnet_cond_scale * controlnet_keep[counter]

        with torch.no_grad():
            down_block_res_samples, mid_block_res_sample = controlnet(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                conditioning_scale=cond_scale,
                guess_mode=False,
                return_dict=False)

        # predict the noise residual
        # print('latent_model_input', latent_model_input)
        # print('text_embeddings', text_embeddings)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample,)[0]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        # print(noise_pred)
        # time()

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        print(counter)
        counter +=1

        # latents = 1 /vae.config.scaling_factor * latents
        # with torch.no_grad():
        #     image = vae.decode(latents).sample

        # print(image.size())
        # image = image_processor.postprocess(image, output_type="pil")[0]
        
        # image.save(args.generated_img_pth)

    latents = 1 /vae.config.scaling_factor * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    print(image.size())
    image = image_processor.postprocess(image, output_type="pil")[0]
    image.save(args.generated_img_pth)


def get_ssim(ori_img_pth, gen_img_pth):
    generated_image = cv2.imread(gen_img_pth, cv2.IMREAD_GRAYSCALE)
    ground_truth_image = cv2.imread(ori_img_pth, cv2.IMREAD_GRAYSCALE)
    height, width = ground_truth_image.shape[:2]
    generated_image_resized = cv2.resize(generated_image, (width, height), interpolation=cv2.INTER_LINEAR)
    ssim_value = ssim(generated_image_resized, ground_truth_image)

    return ssim_value
    

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    generated_img = hey_u_generate_image(args)
    ssim_score = get_ssim(args.input_img_pth, args.generated_img_pth)
    print('SSIM Score::',ssim_score)


