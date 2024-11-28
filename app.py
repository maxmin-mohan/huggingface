import gradio as gr
import numpy as np
import spaces
import torch
from diffusers import DiffusionPipeline
from numpy.random import PCG64DXSM, Generator, SeedSequence
from typing import Tuple, Any

dtype: torch.dtype = torch.bfloat16
device: str = "cuda" if torch.cuda.is_available() else "cpu"


pipe = DiffusionPipeline.from_pretrained("shuttleai/shuttle-3-diffusion", torch_dtype=dtype).to(device)
trigger_word = "CNSTLL"
pipe.load_lora_weights("adirik/flux-cinestill")
# Enable VAE tiling
pipe.vae.enable_tiling()


# Define cinematic aspect ratios
ASPECT_RATIOS = {
    "2.39:1 (Modern Widescreen)": 2.39,
    "2.76:1 (Ultra Panavision 70)": 2.76,
    "3.00:1 (Experimental Ultra-wide)": 3.00,
    "4.00:1 (Polyvision)": 4.00,
    "2.55:1 (CinemaScope)": 2.55,
    "2.20:1 (Todd-AO)": 2.20,
    "2.00:1 (Univisium)": 2.00,
    "2.35:1 (Anamorphic Scope)": 2.35,
    "2.59:1 (MGM Camera 65)": 2.59,
    "1.75:1 (IMAX Digital)": 1.75,
    "1.43:1 (IMAX 70mm)": 1.43,
    "2.40:1 (Modern Anamorphic)": 2.40
}
MAX_SEED = np.iinfo(np.int64).max
MIN_WIDTH = 512
MAX_WIDTH = 3072
STANDARD_WIDTH = 2048
STEP_WIDTH = 8
STYLE_PROMPT = "analog film, high resolution, cinestill 800t, hyperrealistic, widescreen, anamorphic, vignette, bokeh, film grain, dramatic lighting, epic composition, moody, detailed, super wide shot, atmospheric, backlit, soft light,  "

def calculate_height(width: int, aspect_ratio: float) -> int:
    height = int(width / aspect_ratio)
    return (height // 8) * 8

# Pre-calculate height mappings for common widths
HEIGHT_CACHE = {}
for ratio_name, ratio in ASPECT_RATIOS.items():
    HEIGHT_CACHE[ratio_name] = {
        width: calculate_height(width, ratio)
        for width in range(MIN_WIDTH, MAX_WIDTH + 1, STEP_WIDTH)
    }

def validate_aspect_ratio(ratio_name: str) -> float | None:
    return ASPECT_RATIOS.get(ratio_name)

# Replace the single rng instance with a function that creates a fresh generator each time
def get_random_seed() -> int:
    # Create a new generator with a random seed each time
    ss = SeedSequence()
    rng = Generator(PCG64DXSM(ss))
    return int(rng.integers(0, MAX_SEED))

@spaces.GPU()
def infer(
    prompt: str,
    aspect_ratio: str,
    width: int,
    seed: Any = 42,  # Change type hint to Any to handle both string and int inputs
    randomize_seed: bool = False,
    num_inference_steps: int = 4,
    progress: Any = gr.Progress(track_tqdm=True)
) -> Tuple[Any, int]:
    # Prepend style prompt to user input
    FULL_PROMPT = f"{STYLE_PROMPT} {prompt}"
    # print(f"Generating image with prompt: {FULL_PROMPT}")
    
    if randomize_seed:
        seed = get_random_seed()
    else:
        # Convert seed to int if it's a string
        seed = int(seed)
    
    ratio = validate_aspect_ratio(aspect_ratio)
    if ratio is None:
        raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")
        
    generator = torch.Generator().manual_seed(seed)
    height = HEIGHT_CACHE[aspect_ratio][width]
    # Calculate megapixel count
    MEGAPIXEL_COUNT = (width * height) / 1000000
    print(f"Generating {MEGAPIXEL_COUNT} megapixel image.")
    
    image = pipe(
        prompt=FULL_PROMPT,  # Use the combined prompt
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        generator=generator,
        guidance_scale=3.5,
        max_sequence_length=256
    ).images[0]
    return image, seed

examples = [
    # Rocket Car
    [
        "NOT PANORAMIC, NOT MIRRORED. a rocket car going across the bonneville salt flats, the image is blurred to show the immense speed.", # prompt
        "4.00:1 (Polyvision)", # aspect_ratio
        3072, # width   
        23, # seed
        False, # randomize_seed
        4, # num_inference_steps
    ],
    # Taxi Driver
    [
        "This gripping frame captures a close-up of a man, his face illuminated by the harsh red glow of city lights, evoking a mood of unease and introspection. His expression is intense and unreadable, with a hint of brooding menace. The dark, blurred background suggests a bustling urban night, with neon lights flickering faintly, emphasizing the gritty, isolating atmosphere. The contrast between the man’s rugged features and the vibrant red lighting highlights the tension and internal conflict likely central to the scene, immersing the viewer in the character’s psychological state.",  # prompt
        "2.39:1 (Modern Widescreen)",  # aspect_ratio
        2048,  # width
        0,  # seed
        False,  # randomize_seed
        4,  # num_inference_steps
    ],
    # Leon The Professional
    [
        "This tightly framed shot focuses on the reflective lenses of round sun glasses, worn by a figure with weathered skin. The reflections in the glasses reveal a table with cups and hands mid-gesture, suggesting an intense, unseen discussion or ritual taking place. The muted tones and soft lighting enhance the intimate and mysterious mood, drawing attention to the details of the reflections. The perspective feels voyeuristic, as if glimpsing a private moment through the character’s point of view. This evocative close-up emphasizes themes of observation, secrecy, and layered meaning within the narrative.",
        "2.76:1 (Ultra Panavision 70)",
        2048,
        1744078352,
        False,
        4,
    ],
    # Lawrence of Arabia
    [
        "three individuals on camels traversing a vast, sunlit desert. The golden sand stretches endlessly in the foreground, interrupted by the striking presence of dark, rugged mountains in the background, bathed in warm sunlight. The composition emphasizes the isolation and majesty of the desert landscape, with the figures casting long shadows that add depth to the scene. The muted blue sky contrasts beautifully with the earthy tones, creating a balanced and immersive visual. The moment conveys a sense of adventure, introspection, and the timeless allure of the natural world.",
        "2.20:1 (Todd-AO)",
        2048,
        0,
        False,
        4,
    ],
]

css="""
body {
    background-image: url('https://huggingface.co/spaces/takarajordan/CineDiffusion/resolve/main/static/background.webp');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-height: 100vh;
}

gradio-app {
    background: none !important;
}

.gradio-container {
    background-color: white;
}

.dark .gradio-container {
    background-color: #121212;
}

#col-container {
    margin: 0 auto;
    max-width: 100%;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# CineDiffusion 🎥
**CineDiffusion** creates cinema-quality widescreen images at up to **4.2 Megapixels** — *4x higher resolution* than typical AI image generators (1MP). Features authentic cinematic aspect ratios for true widescreen results.
        """)
        
        with gr.Row():
            
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                placeholder="Enter your prompt",
                container=True,
            )
            
            run_button = gr.Button("Run", scale=0)
        
        result = gr.Image(
            label="Result", 
            show_label=False, 
            width="100%",
            type="pil",
            )

        with gr.Row():
            aspect_ratio = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=list(ASPECT_RATIOS.keys()),
                    value="2.39:1 (Modern Widescreen)"
                )
        
        with gr.Accordion("Advanced Settings", open=False):
            
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )
            
            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            
            with gr.Row():
                
                width = gr.Slider(
                    label="Width",
                    minimum=MIN_WIDTH,
                    maximum=MAX_WIDTH,
                    step=STEP_WIDTH,
                    value=STANDARD_WIDTH,
                )
            
            num_inference_steps = gr.Slider(
                label="Number of inference steps",
                minimum=1,
                maximum=50,
                step=1,
                value=4,
            )
        
        gr.Examples(
            examples=examples,
            fn=infer,
            inputs=[prompt, aspect_ratio, width, seed, randomize_seed, num_inference_steps],
            outputs=[result, seed],
            cache_examples=True,
            cache_mode="lazy"
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, aspect_ratio, width, seed, randomize_seed, num_inference_steps],
        outputs=[result, seed]
    )

demo.launch(ssr_mode = False)