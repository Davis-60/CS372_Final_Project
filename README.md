# Background Music Generation Project 
This project uses machine learning to add context aware background music to a video clip.

## Overview (What it Does)
This project is built as a pipeline between 2 ML models. First, a `fine-tuned LLaVA` model analzyes the video clip and provides a text description of music that would best correspond to what is in that video. Second, this text description is used as a prompt for Meta's `Music Gen` model to generate a song of equal length to the video clip. Finally, the AI audio track is combined with the original video clip and saved using `ffmpeg`.

## Quick Start
See SETUP.md for details on creating the Conda Environment. I used DukeCS clusters for development, but project can be run on any machine with a sufficient GPU. 

## Video Links


## Evaluation



    
