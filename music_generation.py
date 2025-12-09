from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

_lazy_model = None

#Lazy loading the model won't help for a script with a single run, but could be useful later on
def lazy_load_model():
    global _lazy_model
    if _lazy_model is None:
        _lazy_model = MusicGen.get_pretrained("facebook/musicgen-large")
        logger.info(f"Model loaded on device: {_lazy_model.device}")
    else:
        logger.info(f"Skipping load, model cached on device: {_lazy_model.device}")
    return _lazy_model

def generate_song(prompt:str, length = 10, name = "current_output"):
    model = lazy_load_model()
    model.set_generation_params(duration=length) 

    logger.info(f"Generating song for prompt: {prompt}")
    wav = model.generate([prompt])

    script_dir = Path(__file__).resolve().parent
    output_path = script_dir /"AI_Songs"/ name

    audio_write(output_path, wav[0].cpu(), model.sample_rate, strategy="loudness")
    logger.info(f"Song saved at {output_path}")


if __name__ == "__main__":
    generate_song( 
    """
    Racing in Toko electronic music
    """, 
    20, "cars")


