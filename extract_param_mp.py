import os
import cv2
import torch
import numpy as np
import torch.multiprocessing as mp
from torch import Tensor
from torch.multiprocessing import Queue, Process
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import traceback
from skimage.transform import estimate_transform, warp
from utils.mediapipe_utils import run_mediapipe
from src import smirk_encoder
from src.smirk_encoder import SmirkEncoder


class FrameExtractor:
    """Extracts frames from videos and prepares them for the 3DMM model."""

    def __init__(self, frame_interval: int = 1, target_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            frame_interval: Extract every nth frame
            target_size: Size to resize frames to (height, width)
        """
        self.frame_interval = frame_interval
        self.input_size = 224
        self.target_size = target_size

    def crop_face(self, frame, landmarks, scale=1.0, image_size=224):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform

    def extract_frames(self, video_path: str) -> tuple[list[Tensor], float]:
        """Extract frames from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (list of frames as numpy arrays, fps of video)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # if frame_count % self.frame_interval == 0:
            kpt_mediapipe = run_mediapipe(frame)

            if kpt_mediapipe is None:
                print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
                # exit()

            kpt_mediapipe = kpt_mediapipe[..., :2]
            tform = self.crop_face(frame, kpt_mediapipe, scale=1.2, image_size=self.input_size)

            cropped_image = warp(frame, tform.inverse, output_shape=self.target_size, preserve_range=True).astype(
                np.uint8)
            # cropped_kpt_mediapipe = np.dot(tform.params,
            #                                np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T).T
            # cropped_kpt_mediapipe = cropped_kpt_mediapipe[:, :2]

            # Convert from BGR to RGB
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = cv2.resize(cropped_image, self.target_size)
            cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            # [3, 224, 224]

            # Resize frame
            # frame = cv2.resize(frame, self.target_size[::-1])  # cv2 expects (width, height)
            frames.append(cropped_image)

            frame_count += 1

        cap.release()
        return frames, fps

    def preprocess_frames(self, frames: List[Tensor]) -> torch.Tensor:
        """Prepare frames for the 3DMM model.

        Args:
            frames: List of frames as numpy arrays

        Returns:
            Tensor of preprocessed frames
        """
        # Stack frames into a batch
        # batch = np.stack(frames, axis=0)
        batch = torch.stack(frames, dim=0)

        # # Convert to float and normalize to [0, 1]
        # batch = batch.astype(np.float32) / 255.0
        #
        # # Normalize using ImageNet stats (if your model expects this)
        # mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        # std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        # batch = (batch - mean) / std
        #
        # # Convert to PyTorch tensor and move channel dimension to position 1
        # batch = torch.from_numpy(batch).permute(0, 3, 1, 2)

        return batch


class Model3DMM:
    """Wrapper for the 3DMM extraction model."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Args:
            model_path: Path to the pretrained model
            device: Device to run the model on
        """
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load the 3DMM extraction model.

        This is a placeholder - replace with your actual model loading code.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model
        """

        checkpoint = torch.load(model_path)
        checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if
                              'smirk_encoder' in k}  # checkpoint includes both smirk_encoder and smirk_generator
        model = SmirkEncoder()
        model.load_state_dict(checkpoint_encoder)  # self.smirk_encoder.load_state_dict(checkpoint_encoder)
        model.to(self.device)
        model.eval()

        # # Example for loading a PyTorch model
        # model = torch.load(model_path, map_location=self.device)
        # model.eval()  # Set to evaluation model

        return model

    @torch.no_grad()  # Disable gradient computation for inference
    def extract_parameters(self, frames_batch: torch.Tensor) -> Dict[str, np.ndarray]:
        """Extract 3DMM parameters from preprocessed frames.

        Args:
            frames_batch: Tensor of preprocessed frames

        Returns:
            Dictionary of 3DMM parameters
        """
        frames_batch = frames_batch.to(self.device)

        # Run inference
        outputs = self.model(frames_batch)

        # Process outputs - this will depend on your model's output format
        # Example output processing:
        parameters = {
            'shape': outputs['shape_params'].cpu().numpy(),
            'expression': outputs['exp_params'].cpu().numpy(),
            'pose': outputs['pose_params'].cpu().numpy(),
            'texture': outputs['tex_params'].cpu().numpy() if 'tex_params' in outputs else None
        }

        return parameters


def inference_worker(model_path: str, input_queue: Queue, output_queue: Queue, gpu_id: int = 0):
    """Worker process that runs the 3DMM model on GPU.

    Args:
        model_path: Path to the model file
        input_queue: Queue for receiving frames
        output_queue: Queue for sending results
        gpu_id: GPU device ID to use
    """
    try:
        # Set the device, default cuda:0
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        # Load the model
        model = Model3DMM(model_path, device)
        # print(f"Inference worker started on {device}")

        while True:
            # Get data from the input queue
            data = input_queue.get()

            # Check for termination signal
            if data is None:
                break

            video_id, frames_batch = data

            # Extract 3DMM parameters
            parameters = model.extract_parameters(frames_batch)

            # Put results in the output queue
            output_queue.put((video_id, parameters))

    except Exception as e:
        print(f"Error in inference worker: {e}")
        traceback.print_exc()
        # Signal that there was an error
        output_queue.put(("ERROR", str(e)))
    finally:
        # Signal that we're done
        output_queue.put(None)


def video_processor(video_path: str, video_id: str, input_queue: Queue,
                    frame_interval: int = 1, target_size: Tuple[int, int] = (224, 224),
                    batch_size: int = 32):
    """Process a video and extract frames for 3DMM parameter extraction.

    Args:
        video_path: Path to the video file
        video_id: Identifier for the video
        input_queue: Queue for sending frames to the inference worker
        frame_interval: Extract every nth frame
        target_size: Size to resize frames to
        batch_size: Number of frames to process at once
    """
    try:
        # Create a frame extractor
        extractor = FrameExtractor(frame_interval, target_size)

        # Extract frames
        frames, fps = extractor.extract_frames(video_path)

        print(f"Extracted {len(frames)} frames from {video_path}")

        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]  # maintain the remainder

            # Preprocess frames
            frames_batch = extractor.preprocess_frames(batch_frames)

            # Put batch in the input queue
            input_queue.put((f"{video_id}_{i // batch_size}", frames_batch))

    except Exception as e:

        # TODO save errors in record_queue?
        ...

        print(f"Error processing video {video_path}: {e}")
        traceback.print_exc()


def result_collector(output_queue: Queue, result_dir: str, expected_workers: int):
    """Collect and save results from the inference worker.

    Args:
        output_queue: Queue for receiving results from the inference worker
        result_dir: Directory to save results to
        expected_workers: Number of workers sending results to this collector
    """
    try:
        # Create the result directory if it doesn't exist
        os.makedirs(result_dir, exist_ok=True)

        workers_done = 0

        while workers_done < expected_workers:
            # Get result from the output queue
            result = output_queue.get()

            # Check for termination signal
            if result is None:
                workers_done += 1
                continue

            # Check for error signal
            if result[0] == "ERROR":
                print(f"Error in worker: {result[1]}")
                continue

            video_id, parameters = result

            # Save the parameters
            output_path = os.path.join(result_dir, f"{video_id}.npy")
            np.save(output_path, **parameters)

            print(f"Saved parameters for {video_id} to {output_path}")

    except Exception as e:
        print(f"Error in result collector: {e}")
        traceback.print_exc()


def main(video_dir: str, model_path: str, result_dir: str,
         num_workers: int = 8, gpu_ids: List[int] = [0],
         frame_interval: int = 1, batch_size: int = 32):
    """Main function to extract 3DMM parameters from videos.

    Args:
        video_dir: Directory containing video files
        model_path: Path to the 3DMM model
        result_dir: Directory to save results to
        num_workers: Number of video inference workers
        gpu_ids: List of GPU IDs to use
        frame_interval: Extract every nth frame
        batch_size: Number of frames to process at once
    """
    # Make sure CUDA is available if GPU IDs are specified
    # if gpu_ids and not torch.cuda.is_available():
    #     print("CUDA not available, falling back to CPU")
    #     gpu_ids = []

    # Find all video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(Path(video_dir).glob(f"*{ext}")))

    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    print(f"Found {len(video_files)} video files")

    # Initialize queues
    input_queues = [Queue() for _ in range(num_workers)]
    # input_queues = [Queue() for _ in range(len(gpu_ids) or 1)]
    output_queue = Queue()
    record_queue = Queue()

    # Start inference workers (one per GPU, or one on CPU if no GPUs)
    inference_processes = []

    for i in range(num_workers):
        # TODO multi-process share same GPU?
        p =  Process(target=inference_worker,
                     args=(model_path, input_queues[i], output_queue))
        p.start()
        inference_processes.append(p)
    # for i, gpu_id in enumerate(gpu_ids) if gpu_ids else [(0, None)]:
    #     p = Process(target=inference_worker,
    #                 args=(model_path, input_queues[i], output_queue),
    #                 kwargs={'gpu_id': gpu_id} if gpu_ids else {})
    #     p.start()
    #     inference_processes.append(p)

    # Start result collector
    collector_process = Process(target=result_collector,
                                args=(output_queue, result_dir, len(inference_processes)))
    collector_process.start()

    # Start video processing workers
    processing_processes = []
    for i, video_path in enumerate(video_files):
        video_id = video_path.stem
        # Distribute videos across available inference workers
        queue_idx = i % len(input_queues)
        # TODO Potential problem? Each process consumes system resources (memory, file descriptors, process IDs).
        #  If too many processes are created simultaneously, it could exhaust system resources.
        p = Process(target=video_processor,
                    args=(str(video_path), video_id, input_queues[queue_idx]),
                    kwargs={'frame_interval': frame_interval, 'batch_size': batch_size})
        p.start()
        processing_processes.append(p)

    # Wait for video processing to complete
    for p in processing_processes:
        p.join()

    # Signal inference workers to terminate
    for q in input_queues:
        q.put(None)

    # Wait for inference workers to complete
    for p in inference_processes:
        p.join()

    # Wait for result collector to complete
    collector_process.join()

    # check record_queue, saving to json/txt
    while not record_queue.empty():
        message = record_queue.get()

        print("record_queue empty")

    print("All processing complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract 3DMM parameters from videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the 3DMM model")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save results to")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of video processing workers")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="GPU IDs to use")
    parser.add_argument("--frame_interval", type=int, default=1, help="Extract every nth frame")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of frames to process at once")

    args = parser.parse_args()

    main(args.video_dir, args.model_path, args.result_dir,
         args.num_workers, args.gpu_ids, args.frame_interval, args.batch_size)
