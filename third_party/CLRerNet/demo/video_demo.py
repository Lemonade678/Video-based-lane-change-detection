# based on https://github.com/open-mmlab/mmdetection/blob/v2.28.0/demo/image_demo.py
# and modified for video processing.
# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
from argparse import ArgumentParser
import atexit
import tempfile
import cv2
import numpy as np
from tqdm import tqdm # >>> ADDED: For a user-friendly progress bar

# Add the project root to sys.path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary libraries from CLRerNet
from mmdet.apis import init_detector
from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes

# Define the target size for the model input as constants
TARGET_WIDTH = 1640
TARGET_HEIGHT = 590

import cv2
import numpy as np

# The __version__ attribute holds the version string
print(f"OpenCV version: {cv2.__version__}")
def prepare_frame(frame, target_width, target_height):
    """
    >>> MODIFIED: This function now takes an in-memory image (numpy array)
    >>> instead of a file path. It resizes the frame to the target size.
    """
    if frame is None:
        return None
    
    h, w, _ = frame.shape
    target_size = (target_width, target_height)
    
    # Resize the frame to the target size required by the model
    if w != target_width or h != target_height:
        prepared_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    else:
        prepared_frame = frame
        
    return prepared_frame
def Clahe_enhance_frame_apply(frame):
    """
    enhance the Contrast and brightness of the frame using CLAHE

    Args_i/o:
        frame (numpy.ndarray): Input frame to be enhanced.
    Returns:
        numpy.ndarray: Enhanced frame.    
    """
    if frame is None:
        return None
    
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge the CLAHE enhanced L channel back with the A and B channels
    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    
    # Convert back to BGR color space
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_frame

def parse_args():
    parser = ArgumentParser()
    # >>> MODIFIED: Changed 'img' to 'video' for clarity
    parser.add_argument('video', help='Video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    # >>> MODIFIED: Help text updated for video output
    parser.add_argument('--out-file', default='result.mp4', help='Path to output video file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args

def main(args):
    # >>> SECTION REBUILT FOR VIDEO PROCESSING <<<

    # 1. Build the model from a config file and a checkpoint file
    # This is done ONCE outside the loop for efficiency.
    print("Initializing CLRerNet model...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print("Model initialized.")

    # 2. Setup video reader and writer
    import os

    video_path = args.video # or the hardcoded path
    print(f"Checking for file at: {video_path}")

    if not os.path.exists(video_path):
        print("ERROR: Video file not found at the specified path!")
    else:
        print("Video file found. Attempting to open with OpenCV...")
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            print("Success: Video opened!")
        else:
            print("ERROR: Video file found, but could not be opened by OpenCV.")

    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or use 'XVID', 'MJPG', etc.
    writer = cv2.VideoWriter(args.out_file, fourcc, fps, (original_width, original_height))
    
    print(f"Processing {frame_count} frames from {args.video}...")
    print(f"Output will be saved to {args.out_file}")

        # 3. Process video frame by frame
    try:
        # Use tqdm for a progress bar
        with tqdm(total=frame_count, desc="Processing Video") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                # 0.enhance the frame using CLAHE
                frame = Clahe_enhance_frame_apply(frame)
                # a. Prepare the current frame
                prepared_frame = prepare_frame(frame, TARGET_WIDTH, TARGET_HEIGHT)
                if prepared_frame is None:
                    continue

                temp_image_path = None  # Initialize path variable
                try:
                    # b. Save the prepared frame to a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_f:
                        temp_image_path = temp_f.name
                        cv2.imwrite(temp_image_path, prepared_frame)

                    # c. Perform inference on the temporary image file
                    src, preds = inference_one_image(model, temp_image_path)

                    # d. Visualize the lanes on the source image
                    dst = visualize_lanes(src, preds, save_path=None)

                    # e. Resize the output image back to original dimensions
                    final_frame = cv2.resize(dst, (original_width, original_height), interpolation=cv2.INTER_AREA)

                    # f. Write the final frame to the output video
                    writer.write(final_frame)

                finally:
                    # g. IMPORTANT: Clean up the temporary file immediately
                    if temp_image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

                # h. Update the progress bar
                pbar.update(1)

    finally:
        # 4. Release everything when finished
        print("\nCleaning up...")
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"Video processing complete. Output saved to {args.out_file}")

if __name__ == '__main__':
    args = parse_args()
    main(args)