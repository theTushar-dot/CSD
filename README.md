# ControlledStableDiffusion(CSD)
This repository contains the solutions for Avataar AI assignment. 


## Repository structure
    .
    ├── depth_images                   #Folder contains the depth images used to control the generation process of stable diffusion   
    ├── generated_images               #Folder contains the generated images
    ├── main.py                        #file consist of code for image generation pipeline
    └── utils.py                       #file consist of help functions


## Environment Settings

- python: 3.9.1
- pyTorch: 2.0.1
- diffusers: 0.29.2
- pillow: 9.4.0
- numpy: 1.25.0
- openCV: 4.10.0
- scikit-image: 0.24.0

## Usage
Arguments discription of main.py to generate the image

	•	--seed: Sets the seed for random number generation to ensure reproducibility of results.
	•	--use_cuda: If specified, the program will utilize GPU acceleration for computations.
	•	--prompt: The text prompt to guide the image generation process.
	•	--num_inference_steps: Specifies the number of inference steps to be used in the generation process.
	•	--height: Sets the height of the generated image in pixels.
	•	--width: Sets the width of the generated image in pixels.
	•	--beta_start: The starting value of the beta parameter for the noise schedule in the diffusion process.
	•	--beta_end: The ending value of the beta parameter for the noise schedule in the diffusion process.
	•	--control_guidance_start: The starting value for control guidance strength.
	•	--control_guidance_end: The ending value for control guidance strength.
	•	--controlnet_con_scale: Scale factors for control net contribution, allowing multiple values for multiple ControlNet inputs.
	•	--guidance_scale: The scale of guidance used in the generation process to steer the output towards the prompt.
	•	--scheduler: Specifies the type of scheduler to use for the generation process.
	•	--input_img_pth: The file path of the input image to be used in the generation process.
	•	--generated_img_pth: The file path where the generated image will be saved.
	•	--control_with_depth: If specified, the generation process will be controlled using a depth image.
	•	--control_with_canny: If specified, the generation process will be controlled using Canny edge detection.
	•	--control_with_normals: If specified, the generation process will be controlled using surface normal information.
	•	--control_with_segment: If specified, the generation process will be controlled using segmentation information.
	•	--use_f16: If specified, the program will use 16-bit floating point precision for faster computation

## General image generation using controlNet

 To generate image using only depth information using LMSDiscreteScheduler and 16-bit floating point precision:

 ```
python main.py --use_cuda --prompt "luxury bedroom interior" --input_img_pth "./depth_images/2.png" --generated_img_pth "./generated_images/2_depth.png" --control_with_depth --num_inference_steps 20  --use_f16
```
![Alt text](./generated_images/2_depth.png)

 To generate image using only canny edge information using LMSDiscreteScheduler and 16-bit floating point precision:

 ```
python main.py --use_cuda --prompt "luxury bedroom interior" --input_img_pth "./depth_images/2.png" --generated_img_pth "./generated_images/2_canny.png" --control_with_canny --num_inference_steps 20  --use_f16
```
![Alt text](./generated_images/2_canny.png)

 To generate image using only surface normals information using LMSDiscreteScheduler and 16-bit floating point precision:

 ```
python main.py --use_cuda --prompt "luxury bedroom interior" --input_img_pth "./depth_images/2.png" --generated_img_pth "./generated_images/2_normal.png" --control_with_normal --num_inference_steps 20  --use_f16
```
![Alt text](./generated_images/2_normal.png)

 To generate image using only segement information using LMSDiscreteScheduler and 16-bit floating point precision:

 ```
python main.py --use_cuda --prompt "luxury bedroom interior" --input_img_pth "./depth_images/2.png" --generated_img_pth "./generated_images/2_segment.png" --control_with_segment --num_inference_steps 20  --use_f16
```
![Alt text](./generated_images/2_segment.png)

##Best generated image

Here, I found the best image generated when using depth and surface normals informatio with conditioning scale of 1.0 and 0.5 respectively with LMSDiscreteScheduler.

1. with 50 inference steps:
inference time = 8.596s

 ```
python main.py --use_cuda --prompt "luxury bedroom interior" --input_img_pth "./depth_images/2.png" --generated_img_pth "./generated_images/2_depth_n_normal_50.png" --control_with_depth control_with_normal  --controlnet_con_scale 1.0 0.5 --num_inference_steps 50  --use_f16
```
![Alt text](./generated_images/2_depth_n_normal_50.png)

2. With 10 inference steps:
inference time = 3.319s
 ```
python main.py --use_cuda --prompt "luxury bedroom interior" --input_img_pth "./depth_images/2.png" --generated_img_pth "./generated_images/2_depth_n_normal_10.png" --control_with_depth control_with_normal  --controlnet_con_scale 1.0 0.5 --num_inference_steps 10  --use_f16
```
![Alt text](./generated_images/2_depth_n_normal_10.png)

I found with more inference steps, i.e., 50, model will try to generate the images with more texture but given the inferenece time. I would contest that for realworld using 10 inference steps will work more good as inference time is more than 60% lower and quality of image remains more or less same.

Just to compare the other combination generated image are shown below(all with --controlnet_con_scale 1.0 0.5 --num_inference_steps 10  --use_f16).
1. Depth and Canny
![Alt text](./generated_images/2_depth_n_canny_10.png)
3. Depth and Segment
![Alt text](./generated_images/2_depth_n_normal_10.png)

The study of impact of Various Scheduler (dinosier) and 16/32-but floating point precision are discussed in last section.


##Generating images of different aspect ration
For this purpose "./depth_images/2_nocrop.png" is used as mentioned in the assignment (all images are genrated using depth information and normal surface information with conditioning scale of 1.0 and 0.5 respectively and 10 inference steps LMSDiscreteScheduler)

1. Generating image with original aspect ratio with 32-bit floating point precision:
Inference Time:  8.313s
![Alt text](./generated_images/2_nocrop_32bit_out.png)

2. Generating image with original aspect ratio with 16-bit floating point precision:
Inference Time:  3.981s
![Alt text](./generated_images/2_nocrop_16_bit_out.png)

2. Generating 1:1 aspect ratio with 16-bit floating point precision:
Inference Time:   3.156s
![Alt text](./generated_images/2_nocrop_1r1_16_bit_out.png)



##Inference time
I found the easist way to reduce the inference time is to use 16-bit floating point precision instead of 32-bit. The choice of schuduler is also quite curcial for inference time.

1. Impact of 16-bit vs 32-bit inference time (all images are generated using LMSDiscreteScheduler depth and normal surface information with conditioning scale of 1.0 and 0.5 respectively and 10 inference):
a. 32 bit
b. 16 bit

2. Impact of various schuduler on inference time and generation quality (all images are generated with depth and normal surface information with conditioning scale of 1.0 and 0.5 respectively):

a. PNDMScheduler: The Pseudo Numerical Methods for Diffusion Models (PNDM) scheduler accelerates diffusion model sampling by integrating pseudo numerical techniques, which balance between stability and speed, enhancing image generation quality.

b. DDIMScheduler: The Denoising Diffusion Implicit Models (DDIM) scheduler offers deterministic and efficient sampling with fewer steps by leveraging implicit noise prediction, allowing for smoother and faster image synthesis in diffusion models.

c. DDPMScheduler: The Denoising Diffusion Probabilistic Models (DDPM) scheduler follows the traditional probabilistic framework of diffusion models, providing robust and stable sampling but typically requires more inference steps for high-quality outputs.

d. LMSDiscreteScheduler: The Linear Multistep Scheduler (LMS) for discrete steps uses linear multistep methods to solve the reverse diffusion process, achieving high-quality image generation with fewer inference steps, making it well-suited for use with ControlNet and Stable Diffusion.

e. HeunDiscreteScheduler: The Heun’s Method Scheduler applies Heun’s method, an improved Euler method, to the discrete diffusion process, offering enhanced accuracy and stability in image generation by correcting for potential errors at each step.

##Some more important findings:
1.  sudden convergence phenomenon
2.  partially breaking the connections between a ControlNet block and the Stable Diffusion model helps in faster convergence.
3.  we need to do two forward passes:
4.  negative_prompt: We can also pass negative prompt to the stable diffusion, model will use the negative prompt embeddings to discrage certain feature as mentioned in the negative prompts. We can provide some very general nagative prompt such as "low res, worst quality, low quality". So, we don't have to give specific nagative prompts for each input and it does not affects the inferenec time as well.
 ![Alt text](./generated_images/2_negative_prompt.png)
7.  Is CPU initialized generator works well?
8.  Token margin is good but for bigger images




 
