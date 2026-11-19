import torch
from PIL import Image, ImageOps
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

#Prepare the sketch

sketch = Image.open("sketches/alien_1.jpeg").convert("L").resize((512, 512))
control = ImageOps.invert(sketch).convert("RGB")
control.save("outputs/alien_1_control.png")

#load the model 
controlnet = ControlNetModel.from_pretrained(
     "lllyasviel/control_v11p_sd15_scribble", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet, torch_dtype=torch.float16).to("cuda")

#Generate the image
prompt = (" a flying saucer UFO with a glass dome, a small cute purple alien inside"
         "waving its hands, bright light beam shining down, night sky above a forest,"
         "oil painting, romantic art style")
image = pipe(prompt, image=control, num_inference_steps=25,
             generator=torch.manual_seed(42)).images[0]
image.save("outputs/first_test.png")
print("done!")


