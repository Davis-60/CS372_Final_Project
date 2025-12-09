import subprocess
import logging
logger = logging.getLogger(__name__)

def mux(video_path, audio_path, output_path):
    logger.info("MUX Video and Audio with ffmpeg")
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",     
        "-map", "1:a:0",     
        "-c:v", "copy",      
        "-c:a", "aac",       
        "-shortest",         
        output_path
    ]
    subprocess.run(cmd, check=True)

