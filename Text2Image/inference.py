import os
import torch
from diffusers import StableDiffusionPipeline
from sd_pipeline import DiffusionPipeline
# Load the pre-trained Stable Diffusion model (v1.4)
model_id = "/home/rt/data/SD_playground/CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
# pipe = pipe.to(device)

# def generate_image(prompt, num_inference_steps=50, guidance_scale=7.5):
#     # Generate image
#     with torch.no_grad():
#         image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
#     return image


pipeline = DiffusionPipeline(
        pretrained_model_name=model_id,
        pretrained_model_revision='main',
        use_lora=True,
    )
pipeline.sd_pipeline.safety_checker = None
pipeline.load_checkpoint(
    models=[pipeline.unet],
    input_dir=f"./checkpoints/alignprop_1736903427/checkpoints/checkpoint_0", #/home/rt/data/diff_LR/Text2Image/checkpoints/alignprop_1736903427/checkpoints/checkpoint_15
)
pipeline.sd_pipeline = pipeline.sd_pipeline.to(device)

if __name__ == "__main__":
    save_dir = "./latent_images"
    os.makedirs(save_dir, exist_ok=True)
    prompt = "A artistic picture of swift Cheetah"
    output = pipeline.rgb_with_grad(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=1,
        backprop_strategy='None',
        backprop_kwargs=None,
        output_type="pil",
        cache_latents=True,
    )  
    image = output.images[0]
    all_latent = output.latents

    for i,item in enumerate(all_latent):
        item[0].save(save_dir+f"/latent{i}.png")

    # image = generate_image(prompt)
    print(image)
    image.save("generated_image.png")
    print("Image saved as generated_image.png")
    
