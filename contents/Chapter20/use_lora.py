import torch
from diffusers import (
    StableDiffusionPipeline,
    EulerAncestralDiscreteScheduler,  # or DPMSolverMultistepScheduler
)

model_id = "stabilityai/stable-diffusion-2"

# Choose your device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Pick your scheduler
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    model_id, 
    subfolder="scheduler"
)

# For DPMSolver, use:
#from diffusers import DPMSolverMultistepScheduler
#scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", algorithm_type="dpmsolver++")

# 2. Load the pipeline with the chosen scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# 3. (Optional) Load LoRA weights
pipe.load_lora_weights("lmoroney/finetuned-misato-sd2")

# 4. Define prompts and parameters
prompt = (
    "Fashion editorial portrait of <lm-misato-lora>, high tech sunglasses, cinematic lighting by Alain Laboile, backlit, shy yet confident expression, lens flares, high-end DSLR at f/1.8, hyperdetailed eyes, professional retouching, 8k, shallow depth of field, volumetric lighting, cinematic angle, dynamic composition"
)
negative_prompt = (
    "low quality, grainy, text watermark, cartoonish, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, "
    "extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), "
    "disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
)

num_inference_steps = 30
guidance_scale = 8.0
width = 512
height = 512
seed = 12345

# 5. Create a generator for reproducible results
generator = torch.Generator(device=device).manual_seed(seed)

# 6. Run the pipeline
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]

# 7. Save the result
image.save("lora-with-negative.png")
