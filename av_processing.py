import subprocess
import logging
logger = logging.getLogger(__name__)


def mux(video_path, audio_path, output_path):
    logger.info("MUX Video and Audio with ffmpeg")
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",     # take video from input 0
        "-map", "1:a:0",     # take audio from input 1 (your WAV)
        "-c:v", "copy",      # don't re-encode video
        "-c:a", "aac",       # encode WAV to AAC
        "-shortest",         # match shortest stream
        output_path
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    mux("CS372_Final_Project/Input_Videos/horses.mp4", "CS372_Final_Project/AI_Songs/horses.wav", "CS372_Final_Project/Output_Videos/combined_horses.mp4")
