import argparse
import torch
import time
import numpy as np
import cv2
from tracker import AnimalTracker
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--batch', type=int, required=True, help='Number of frames to be processed in parallel')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output CSV')
    parser.add_argument('--generate_video', action='store_true', help='Generate a tracked video (batch by batch)')
    args = parser.parse_args()

    start_time = time.time()  # Start timer

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        import warnings
        warnings.warn("You are using CPU as your device. This code is not optimized for CPU.")

    # Load video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)

    tracker = AnimalTracker(device=device, batch_size=args.batch)
    com_array = np.empty((0,2))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {frame_count}")

    batch_idx = 1

    while True:
        # Read a batch of frames
        batch_frames = []
        for _ in range(args.batch):
            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
        if not batch_frames:
            break
        
        # Process frames
        com = tracker.process_frame(batch_frames).detach().cpu().numpy()

        # Accumulate the results in an np array
        com_array = np.append(com_array, com, axis=0)

        print(f"Processed Frames: {batch_idx} * {args.batch}")
        batch_idx += 1

        if args.generate_video:
            generate_tracked_video(batch_frames, com, f"tracking_batch{batch_idx}.mp4", fps=fps)

    cap.release()
    save_results(com_array, args.output)
    
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s")  # Output formatted time

if __name__ == '__main__':
    main()