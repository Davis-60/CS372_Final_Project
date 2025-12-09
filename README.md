# Background Music Generation Project 
This project uses machine learning to add context aware background music to a video clip.

## Overview (What it Does)
This project is built as a pipeline between 2 ML models. First, a `fine-tuned LLaVA` model analzyes the video clip and provides a text description of music that would best correspond to what is in that video. Second, this text description is used as a prompt for Meta's `Music Gen` model to generate a song of equal length to the video clip. Finally, the AI audio track is combined with the original video clip and saved using `ffmpeg`.

## Quick Start
1. See SETUP.md for details on creating the Conda Environment. 
2. Add your initial video clip to data/Input_Videos
3. Run generation in src/combined_generation.py
4. Download Output Video from src/Output_Videos

## Video Links
A demo and walkthrough video are available at [demo video](videos/demo.mp4) and [walkthrough video](videos/technical_walkthrough.mp4).

## Evaluation
Qualitative Evaluation perfomed on the 4 videos clips stored in Output_Videos. The fine tuned LLaVA model generated descriptions describing plausable music for each clip. For example, this was the description for the `horses.mp4` input video:

    The music that would best compliment this video is a beautiful, serene and gentle melody. The melody should have a calm and soothing nature, and it should be able to evoke a sense of tranquility and peace. The melody should be accompanied by light and lively acoustic guitar strumming, and the rhythm should be steady and steady, mimicking the calm and steady pace of the horses' movement in the video.

Qualitatively the final outputs' background music qualitatvely meets expecations based on the visual context of each clip. 



    
