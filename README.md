Bring Sketches to Life : Alien meets Cat

I love space and aliens and I wanted to tell a tiny story of an Alien that arrives to the Earth and adopts a cat that jumped in to his spaceship. I used Stable Diffusion 1.5 + ControlNet.

Sketch strength comparison
How it works :

hand-drawn sketch ──► preprocess.py ──► control image ──┐
(phone photo)         (OpenCV)          white lines     │
                                         on black       ▼
text prompt ──► CLIP text encoder ──────────────►  U-Net + ControlNet  ──►  VAE decoder  ──►  image
(scene + style)                                    (25 denoising steps)

1.- Preprocessing (preprocess.py) _
grayscale -> resize to 512x512 -> Gaussian blur (removes paper grain)-> adaptive Gaussian threshold, inverted ->dilatation. The result is white lines on a black background, the format ControlNet scribble was trained on. 

2.- Generatio (generate.py):Stable Diffusion 1.5 starts from random noise in a 64×64 latent space and removes it step by step, guided by the CLIP embedding of the prompt. ControlNet reads the control image and adds corrections inside the U-Net at every step, so the sketch decides where things are and the prompt decides what they look like. controlnet_conditioning_scale sets how strongly the sketch is followed (0 = text only).

3.- Story Consistency : The alien and the littel cat are described once (ALIEN and CAT in epxeriments.py ) and reused in every scene prompt with the same seed so my characters stay similars across the story.

4.-  Evaluation (metrics.py): Canny edges are detected in the generated image and compared with the sketch strokes (4 px tolerance, using a distance transform).
Recall: share of sketch line pixels found in the image (structure preservation).
Precision: share of image edges lying on the sketch (low = many details added by the model).

Experiments
Experiment	What varies	Fixed
Sketch strength	controlnet_conditioning_scale = 0.0, 0.5, 1.0, 1.3	style = oil, seed = 42
Style	oil painting, photorealistic, watercolor	scale = 1.0, seed = 42

Results

Sketch recall for alien_1 (scale 0.0 = text-only baseline):

Scale	Recall	Precision
0.0	0.34	0.21
0.5	0.45	0.27
1.0	0.70	0.53
1.3	0.69	0.77

Recall vs scale

Main findings

The sketch clearly controls the result: recall doubles from 0.34 (text only) to 0.70 at scale 1.0.
Beyond 1.0, recall saturates while precision rises: stronger conditioning removes added detail instead of preserving more structure. Scale 1.0 is the best balance.
The prompt must describe what is drawn: with a mismatched prompt, the UFO shape was kept but reinterpreted as a creature.
Small details (the alien's hands) are often lost at 512×512.

Installation

Tested on: Lenovo Legion Pro, NVIDIA GeForce RTX 5070 Laptop GPU (12 GB VRAM), 32 GB RAM, Windows 11, Python 3.11, PyTorch 2.11 + CUDA 12.8.


Main settings (for reproducibility)
Setting	Value
Base model	stable-diffusion-v1-5/stable-diffusion-v1-5
ControlNet	lllyasviel/control_v11p_sd15_scribble
Scheduler	UniPC multistep, 25 steps
Resolution	512 × 512
Guidance scale	7.5
Conditioning scale	1.0 (varied in the scale experiment)
Seed	42
Precision	float16 on CUDA
Negative prompt	blurry, low quality, deformed, distorted, watermark, text, error
Time per image	2.75s on the RTX 5070 

sketches/          my input drawings
outputs/           generated images (one folder per sketch), grids, CSV results
first_test.py      first prototype (one sketch → one image)
preprocess.py      sketch → control image
generate.py        load the pipeline + generate one image
experiments.py     run all experiments on all sketches
metrics.py         sketch recall / precision
analyze.py         metrics table, comparison grids, plot
requirements.txt   Python packages

References
Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, CVPR 2022.
Zhang, Rao & Agrawala, Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet), ICCV 2023.
Models: stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M licence), lllyasviel/control_v11p_sd15_scribble.
Libraries: diffusers, transformers, PyTorch, OpenCV, Pillow, matplotlib.
Sketches: drawn by me (Astrid Miranda).
AI assistance: I used Claude (Anthropic) as a tutor and coding assistant to explain concepts