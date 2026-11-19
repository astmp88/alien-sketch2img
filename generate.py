import time 
import torch
from diffusers import ControlNetModel, StableDiffusionPipeline, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

NEGATIVE = "blurry, low quality, deformed, distorted, watermark, text, error"

def load_pipeline(controlnet_name = "lllyasviel/control_v11p_sd15_scribble"):
    controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet=controlnet, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

def generate(pipe, control, prompt, scale=1.0, guidance=7.5, steps=25, seed=42):
    start = time.time()
    image = pipe(
        prompt,
        image=control,
        negative_prompt=NEGATIVE,
        num_inference_steps=steps,
        guidance_scale=guidance,
        controlnet_conditioning_scale=scale,
        generator=torch.manual_seed(seed)
    ).images[0]
    seconds = time.time() - start
    print(f"Generation took {seconds:.2f} seconds")
    return image, seconds

if __name__ == "__main__":
    from preprocess import preprocess_sketch

    pipe = load_pipeline()
    control = preprocess_sketch("sketches/alien_1.jpeg")
    prompt = ("a flying saucer UFO with a glass dome, a cute purple alien with two antennae "
              "and a smiling face waving both hands inside the dome, bright light beam shining down, "
              "night sky above a forest, oil painting, romantic art style")

    for scale in [0.0, 0.5, 1.0, 1.3]:
        image, seconds = generate(pipe, control, prompt, scale=scale)
        image.save(f"outputs/alien_1_scale_{scale}.png")