## Setup Notes
### Environment
Runs inside the `unified-env` Conda virtual environment.

### Hardware Requirements  
Requires GPU with more than 20GB of VRAM and CUDA support.  
An RTX a5000 on the Duke CS Clusters was used during development.


## Conda Environment Setup
### 1. Create the conda envirnonment with most dependencies
    conda env create -f environment.yml

### 2. Activate the environment
    conda activate unified-env

### 3. Install final tricky dependencies
    python -m pip install "xformers==0.0.27.post2" "peft==0.6.0" audiocraft --no-deps
