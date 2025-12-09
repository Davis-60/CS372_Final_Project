import av
import torch
import numpy as np
from pathlib import Path
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import logging
from peft import PeftModel

logger = logging.getLogger(__name__)
source_dir = Path(__file__).resolve().parent.parent
lora_path =  source_dir / "models" / "llava_video_checkpoints" / "checkpoint-249"
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

# Loading Base LLaVa Model
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

# Adding LORA Adapter
print(f"Loading LoRA Adapter from {lora_path}...")
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# Helper function to load frames from video
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# Helper function to return video duration in seconds
def get_video_duration(video_path: str):
    container = av.open(video_path)
    video = container.streams.video[0]
    time = float(video.duration * video.time_base)
    logger.info(f"Video at {video_path} is {time}s")
    return time


def analyze_video(video_path: str, prompt_text: str = "Describe the music that would best compliment this video.") -> str:

    logger.info(f"Analyzing Video at: {video_path}")

    prompt = processor.apply_chat_template( [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt_text},
                ],
        },
    ], add_generation_prompt=True)

    # Sampling 16 frames from video
    with av.open(video_path) as container:
        video = container.streams.video[0]
        indices = np.arange(0, video.frames, video.frames / 16).astype(int)
        clips = read_video_pyav(container, indices)

    # Processing / Tokenizing inputs
    inputs_video = processor(
        text=prompt, 
        videos=clips, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)

    # Generating Description
    output = model.generate(
            **inputs_video, 
            max_new_tokens=100, 
            do_sample=True,         
            temperature = .7,
        )

    # Decoding Output
    generated_ids = output[0][len(inputs_video.input_ids[0]):]
    final_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
    logger.info(f"Video Description: {final_output}")
    
    return final_output





