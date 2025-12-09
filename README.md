## Conda Environment Setup
    conda env create -f CS372_Final_Project/unified_env.yml
    conda activate unified-env
    python -m pip install "xformers==0.0.27.post2" "peft==0.6.0" audiocraft --no-deps


## Aquiring Testing Videos

    `yt-dlp -f "bv*[height>=720]+ba"`

    downscale with `ffmpeg` if needed

        ffmpeg -i Desktop/Beautiful\ Rescue\ Horses\ Run\ By\ \[jMe2QUZhOxQ\].mkv  -vf scale=-1:720 -c:v libx264 -crf 18 -preset slow -c:a aac horses.mp4

## Requesting GPU (Extra Slurm Argument)

    --gres=gpu:a5000
    
