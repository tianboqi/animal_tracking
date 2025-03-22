# Animal Tracking using GPU

This project provides a PyTorch-based pipeline for tracking animal movements in an arena using GPU acceleration. It applies background subtraction, morphological operations, and center of mass detection to identify and analyze the animal's trajectory.

## Features
- GPU-accelerated video processing using PyTorch.
- Especially optimized for low-contrast videos.
  
## Installation
```bash
# Clone the repository
git clone https://github.com/username/animal_tracking.git
cd animal_tracking

# Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. Place your video file in the `video/` folder.
2. Run the pipeline:
```bash
python src/main.py --video video/example.mp4 --batch 2000 --output results.csv
```
3. Adjust the parameters in `src/main.py` and `src/tracker.py`if needed.

## Parameters
- `--video`: Path to the input video.
- `--batch`: Batch size for parallel processing.
- `--output`: Path to save the output CSV with center of mass data.
- `--generate_video`: Whether you want to generate videos with overlaid center-of-mass (batch by batch)

## Results
Output CSV includes the following columns:
- `frame_number`: Frame index.
- `x_com`: X-coordinate of the center of mass.
- `y_com`: Y-coordinate of the center of mass.

## Contact
For questions or contributions, feel free to open an issue or submit a pull request.

