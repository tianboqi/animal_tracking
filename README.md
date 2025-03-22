# Animal Tracking using GPU

This project provides a PyTorch-based pipeline for tracking animal movements in an arena using GPU acceleration. 
Designed to handle long video recordings, it provides robust performance for large-scale behavioral analysis.

## Features
- GPU-accelerated video processing using PyTorch.
- Especially optimized for low-contrast videos.
- Faster than [DeepLabCut](https://www.mackenziemathislab.org/deeplabcut) or [SLEAP](https://sleap.ai/) and no training needed if you <ins>only</ins> need center of mass.
  
## Installation
```bash
# Clone the repository
git clone https://github.com/tianboqi/animal_tracking.git
cd animal_tracking

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. Place your video file in the `video/` folder.
2. Adjust the parameters in `src/main.py` and `src/tracker.py` if needed.
3. Run the pipeline:
```bash
python src/main.py --video video/example.mp4 --batch 2000 --output results.csv
```

## Parameters
- `--video`: Path to the input video.
- `--batch`: Batch size for parallel processing. This should be determined by your GPU memory.
- `--output`: Path to save the output CSV with center of mass data.
- `--generate_video`: Whether you want to generate videos with overlaid center-of-mass (batch by batch).

## Output
The output CSV includes the following columns:
- `frame_number`: Frame index.
- `x_com`: X-coordinate of the center of mass.
- `y_com`: Y-coordinate of the center of mass.

With `--generate_video`, there will be videos generated with the tracked center of mass (a red dot) for each batch of frame processed.

## Contact
For questions or contributions, feel free to open an issue or submit a pull request.

