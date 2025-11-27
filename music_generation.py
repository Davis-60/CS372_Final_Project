import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from pathlib import Path

_lazy_model = None

#Lazy loading the model won't help for a script with a single run, but could be useful later on
def lazy_load_model():
    global _lazy_model
    if _lazy_model is None:
        _lazy_model = MusicGen.get_pretrained("facebook/musicgen-large")
        _lazy_model.set_generation_params(duration=20) 
        print(f"Model loaded on device: {_lazy_model.device}")
    else:
        print(f"Skipping load, model cached on device: {_lazy_model.device}")
    return _lazy_model

def generate_song(model, prompt:str, name = "current_output"):
    print(f"Generating song for prompt: {prompt}")
    wav = model.generate([prompt])

    script_dir = Path(__file__).resolve().parent
    output_path = script_dir /"AI_Songs"/ name

    audio_write(output_path, wav[0].cpu(), model.sample_rate, strategy="loudness")
    print("Song saved\n")


if __name__ == "__main__":
    model = lazy_load_model()
    generate_song(model, "Psyco Song", "psyco")
    #generate_song(model, "Hipster 70s song", "hispter")


