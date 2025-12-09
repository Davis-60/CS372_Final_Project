from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def generate_song(prompt:str, length = 10, name = "current_output") -> None:
    model = MusicGen.get_pretrained("facebook/musicgen-large")
    model.set_generation_params(duration=length) 

    logger.info(f"Generating song for prompt: {prompt}")
    wav = model.generate([prompt])

    script_dir = Path(__file__).resolve().parent
    output_path = script_dir /"AI_Songs"/ name

    audio_write(output_path, wav[0].cpu(), model.sample_rate, strategy="loudness")
    logger.info(f"Song saved at {output_path}")
    return



