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
Arguments discription to generate the image

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
	•	--controlnet_con_scale: Scale factors for control net contribution, allowing multiple values.
	•	--guidance_scale: The scale of guidance used in the generation process to steer the output towards the prompt.
	•	--scheduler: Specifies the type of scheduler to use for the generation process.
	•	--input_img_pth: The file path of the input image to be used in the generation process.
	•	--generated_img_pth: The file path where the generated image will be saved.
	•	--control_with_depth: If specified, the generation process will be controlled using a depth image.
	•	--control_with_canny: If specified, the generation process will be controlled using Canny edge detection.
	•	--control_with_normals: If specified, the generation process will be controlled using surface normal information.
	•	--control_with_segment: If specified, the generation process will be controlled using segmentation information.
	•	--use_f16: If specified, the program will use 16-bit floating point precision for faster computation
