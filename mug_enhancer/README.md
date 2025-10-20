# MUG-V Video Enhancer

## Environment Setup

If you have already set up the environment following the main `README.md`, you can skip this section as all necessary dependencies are already installed. However, if you wish to run the video enhancer independently, follow the steps below to set up the environment specifically for `mug_enhancer`.

### Prerequisites

- **Python** ≥ 3.8 (tested with 3.10)  
- **CUDA** 12.1  

### Install Dependencies

```bash
# Create a new conda environment
conda create -n enhancer python=3.10 -y
conda activate enhancer

# Install required packages
pip install torch>=2.4.1 torchvision>=0.19.1 torchmetrics==1.6.2 \
    einops lightning==2.5.0.post0 pandas diffsynth==1.1.2 \
    peft safetensors pillow
```

## Running the Enhancer

1. **Prepare Checkpoints:**
   Download the necessary checkpoints and place them in the `../pretrained_ckpt/` directory.

   ```bash
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ../pretrained_ckpt/Wan2.1-T2V-1.3B
   ```

2. **Run the Script:**

   Execute the following command to enhance your video:

   ```bash
   ./run.sh
   ```

## Validation Dataset Format

The validation dataset should be a CSV file with the following columns:

- `file_name`: Path or name of the video file
- `text`: Descriptive tags or captions

**Example:**

```csv
file_name,text
sample_000001_video.mp4,"High quality video, HD, clear, motion, cinema"
sample_000002_video.mp4,"High quality video, HD, clear, motion, cinema"
```

