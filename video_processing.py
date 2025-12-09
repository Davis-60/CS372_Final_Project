import av
import torch
import numpy as np
from pathlib import Path
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import logging
from peft import PeftModel

script_dir = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

# Later checkpoints demonstrated overfitting
ADAPTER_PATH =  script_dir / "Lora_Files" / "llava_video_checkpoints" / "checkpoint-249"


model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

# 2. LOAD & MERGE LORA ADAPTER
print(f"Loading LoRA Adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
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


def get_video_duration(video_path: str):
    container = av.open(video_path)
    video = container.streams.video[0]
    time = float(video.duration * video.time_base)
    logger.info(f"Video at {video_path} is {time}s")
    return time


def analyze_video(
    video_path: str, prompt_text: str = "Describe the music in this video."
) -> str:
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

    # 2. VIDEO LOADING: Use 'with' to auto-close the file
    with av.open(video_path) as container:
        video = container.streams.video[0]
        # Sample 32 frames uniformly
        indices = np.arange(0, video.frames, video.frames / 32).astype(int)
        clips = read_video_pyav(container, indices)

    # 3. TOKENIZE
    inputs_video = processor(
        text=prompt, 
        videos=clips, 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)

    # 4. GENERATE
    output = model.generate(
            **inputs_video, 
            max_new_tokens=100, 
            do_sample=True,         # Turn OFF sampling (make it deterministic)
            temperature = .7,
        )

    # 5. DECODE
    generated_ids = output[0][len(inputs_video.input_ids[0]):]
    final_output = processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    logger.info(f"Video Description: {final_output}")
    return final_output


if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,
)
    video_path = script_dir / "Input_Videos" / "horses.mp4"
    output = analyze_video(video_path,)
    video_path = script_dir / "Input_Videos" / "dog_show.mp4"
    output = analyze_video(video_path,)
    video_path = script_dir / "Input_Videos" / "tokyo_cars.mp4"
    output = analyze_video(video_path,)


