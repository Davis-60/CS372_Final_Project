from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import logging
logger = logging.getLogger(__name__)

def generate_song(prompt:str, output_path: str, length = 10, ) -> None:
    model = MusicGen.get_pretrained("facebook/musicgen-large")
    model.set_generation_params(duration=length) 

    logger.info(f"Generating song for prompt: {prompt}")
    wav = model.generate([prompt])

    audio_write(output_path, wav[0].cpu(), model.sample_rate, strategy="loudness")
    logger.info(f"Song saved at {output_path}.wav")
    return



