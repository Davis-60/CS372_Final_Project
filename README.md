## General Environment Info

If there are problems with a misconfigured chanel run `unset CONDA_CHANNELS`

Keep seperate environments for the two models
    

## Music Env
conda env create -f CS372_Final_Project/music_env.yml
conda activate final-project-env
python -m pip install "xformers==0.0.22.post7" audiocraft --no-deps

## Unified Environment (I don't have this working yet)
conda env create -f CS372_Final_Project/unified_env.yml
conda activate unified-env
python -m pip install "xformers==0.0.27.post2" audiocraft bitsandbytes --no-deps


## Aquiring Testing Videos

    `yt-dlp -f "bv*[height>=720]+ba"`

    downscale with `ffmpeg` if needed
    
