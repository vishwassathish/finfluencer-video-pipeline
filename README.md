# Finfluencer-analysis

Analyze the credibility of financial influencers on tiktok, youtube and other platforms

## 1. Install pip packages
- Feel free to create a new conda environment if you prefer and install the necessary packages:
```bash
conda create -n "finfluencer" python=3.12.2 ipython
conda activate finfluencer
pip install -r ./GroundingDINO/requirements.txt
```

## 2. Install GroundingDino and Download model weights
- I have included the GroundingDino repo in the codebase so only weights need to be downloaded. This is the Open world object detector we will be using in the pipeline. Original repository [here](https://github.com/IDEA-Research/GroundingDINO).
- Make sure the `$CUDA_HOME` variable is set to the right path. The [official repository](https://github.com/IDEA-Research/GroundingDINO) has more instructions for this step.
- Run the below commands from the project directory:
```bash
# Install the repo
cd GroundingDINO/
pip install -e .

# Download the weights
mkdir .weights
cd .weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

## 3. Mount Google Drive folder from a remote machine
Follow [this](https://ucr-research-computing.github.io/Knowledge_Base/how_to_mount_google_drive.html) link.