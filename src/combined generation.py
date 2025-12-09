from music_generation import generate_song
from pathlib import Path
from video_processing import analyze_video, get_video_duration
from av_processing import mux
import logging
import torch
logger = logging.getLogger(__name__)

source_dir = Path(__file__).resolve().parent.parent

def e2e_generation(video_name:str, output_name:str) -> None:
    video_path = source_dir / "data" / "Input_Videos" / f"{video_name}.mp4"
    song_path = source_dir / "data" / "AI_Songs" / f"{output_name}"
    output_path = source_dir / "data" / "Output_Videos" / f"{output_name}.mp4"

    # First using LLava to generate a description of the video
    description = analyze_video(video_path)
    #Clearing torch
    torch.cuda.empty_cache()
    # Second using MusicGen-Large to generate a song based on the video description
    duration = get_video_duration(video_path)
    generate_song(description, song_path, duration)
    # Finally combining the video and song into an output video
    mux(video_path, f"{song_path}.wav", output_path)
    logger.info(f"Generation complete, output video saved at {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,)
    # MODIFY INPUT AND OUTPUT NAMES HERE
    e2e_generation("nyc", "nyc")
    