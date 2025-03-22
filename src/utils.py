import numpy as np
import pandas as pd
import cv2

def save_results(com_array, output_path):
    if len(com_array) == 0:
        print("No results to save.")
        return
    
    df = pd.DataFrame(com_array, columns=['x_com', 'y_com'])
    df['frame_number'] = np.arange(len(com_array))
    df = df.reindex(columns=['frame_number', 'x_com', 'y_com'])

    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def generate_tracked_video(batch_frames, com_array, output_video_path, fps):
    if len(batch_frames) != len(com_array):
        print(f"Error: Number of frames and COM points must be equal. There are {len(batch_frames)} frames and {len(com_array)} COMs.")
        return

    frame_height, frame_width = batch_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    for i, (frame, (x_com, y_com)) in enumerate(zip(batch_frames, com_array)):
        if not np.isnan(x_com) and not np.isnan(y_com):
            cv2.circle(frame, (int(y_com), int(x_com)), 5, (0, 0, 255), -1)
        out.write(frame)
    
    out.release()
    print(f"Tracked video saved to {output_video_path}")