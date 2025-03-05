import os
import cv2
import torch
import numpy as np
import torch.multiprocessing as mp
from numpy import ndarray
from torch import Tensor
from torch.multiprocessing import Queue, Process
import time
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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
        self.target_size = target_size
        self.face_net = self.load_face_net()

    def load_face_net(self):
        self.face_detector_model = "models/opencv_face_detector_uint8.pb"
        self.face_detector_config = "models/opencv_face_detector.pbtxt"

        # Check if models exist
        if not os.path.exists(self.face_detector_model) or not os.path.exists(self.face_detector_config):
            print(f"Warning: Face detector models not found at {self.face_detector_model}")
            print("Please download the OpenCV DNN face detector models.")
        else:
            # Initialize the face detector
            # self.face_net = cv2.dnn.readNetFromTensorflow(self.face_detector_model, self.face_detector_config)
            self.face_net = cv2.dnn.readNet(self.face_detector_model, self.face_detector_config)
        return self.face_net

    def crop_face(self, frame, landmarks, scale: float = 1.0, image_size: Tuple[int, int] = (224, 224)):
        print("cropping face ...")
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

        size = int(old_size * scale)
        print(f"old_size: {old_size}, size: {size}")

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_size[0] - 1], [image_size[1] - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform

    def detect_face_opencv(self, frame):
        """Detect faces using OpenCV DNN module"""
        # Implementation of our detect_face_opencv function from above
        height, width = frame.shape[:2]

        # Prepare the frame for the neural network
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)

        # Set the input to the network
        self.face_net.setInput(blob)

        # Run forward pass and get detections
        detections = self.face_net.forward()

        # Process detections
        best_detection = None
        best_confidence = 0.5  # Minimum confidence threshold

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > best_confidence:
                # Get face box coordinates
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)

                # Make sure the face is within the image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                if x1 >= x2 or y1 >= y2:
                    continue

                best_detection = (x1, y1, x2, y2)
                best_confidence = confidence

        if best_detection is None:
            return None

        # Extract the face ROI
        x1, y1, x2, y2 = best_detection

        # Create approximate landmarks based on face geometry
        landmarks = []
        rows, cols = 20, 20  # 400 points total
        for row in range(rows):
            for col in range(cols):
                x = x1 + (x2 - x1) * col / (cols - 1)
                y = y1 + (y2 - y1) * row / (rows - 1)
                z = 0  # No depth information with basic detection
                landmarks.append([x, y, z])

        return np.array(landmarks)

    def extract_frames(self, video_path: str) -> tuple[list[Tensor], float]:
        """Extract frames from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (list of frames as numpy arrays, fps of video)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # detector = self.load_detector()
        # print("detector loaded ...")

        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"video_path: {video_path}, num_frames: {num_frames}, fps: {fps}")

        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            print("Detecting face...")
            # Use our OpenCV detector instead of MediaPipe
            kpt_mediapipe = self.detect_face_opencv(frame)

            if kpt_mediapipe is None:
                print(f"No face detected at frame {frame_count}")
                frame_count += 1
                continue

            kpt_mediapipe = kpt_mediapipe[..., :2]
            tform = self.crop_face(frame, kpt_mediapipe, scale=1.2, image_size=self.target_size)
            print("tform finished ...")

            cropped_image = warp(frame, tform.inverse, output_shape=self.target_size, preserve_range=True).astype(
                np.uint8)
            # cropped_kpt_mediapipe = np.dot(tform.params,
            #                                np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T).T
            # cropped_kpt_mediapipe = cropped_kpt_mediapipe[:, :2]

            # Convert from BGR to RGB
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = cv2.resize(cropped_image, self.target_size)
            cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).float() / 255.0
            # [3, 224, 224]
            print("crop finished ...")

            # Resize frame
            # frame = cv2.resize(frame, self.target_size[::-1])  # cv2 expects (width, height)
            frames.append(cropped_image)

            frame_count += 1
            print(f"frame_count: {frame_count} / {num_frames}", end="\r")

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
    def extract_parameters(self, frames_batch: torch.Tensor) -> ndarray:
        """Extract 3DMM parameters from preprocessed frames.

        Args:
            frames_batch: Tensor of preprocessed frames

        Returns:
            Dictionary of 3DMM parameters
        """
        frames_batch = frames_batch.to(self.device)

        # Run inference
        outputs = self.model(frames_batch)
        expression = outputs['expression_params'].cpu()
        jaw = outputs['jaw_params'].cpu()
        pose = outputs['pose_params'].cpu()
        parameters = torch.cat((expression, jaw, pose), dim=-1).numpy()

        # Process outputs - this will depend on your model's output format
        # Example output processing:
        # parameters = {
        #     'expression': outputs['expression_params'].cpu().numpy(),
        #     'jaw': outputs['jaw_params'].cpu().numpy(),
        #     'pose': outputs['pose_params'].cpu().numpy(),
        # }

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


def video_processor_worker(worker_id, video_paths, input_queue, frame_interval=1, batch_size=32,
                           target_size: Tuple[int, int] = (224, 224)):
    """Process a subset of videos and send frames to the inference queue"""

    extractor = FrameExtractor(frame_interval, target_size)
    print(f"extractor loaded for worker {worker_id}")

    for video_path in video_paths:
        video_id = video_path.stem
        try:
            # Extract frames
            frames, fps = extractor.extract_frames(str(video_path))
            print(f"Worker {worker_id}: Extracted {len(frames)} frames from {video_path}")

            # Process frames in batches
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                frames_batch = extractor.preprocess_frames(batch_frames)
                print("putting batch in queue")
                input_queue.put((f"{video_id}_{i // batch_size}", frames_batch))
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            traceback.print_exc()

    # Signal completion
    return f"Worker {worker_id} completed processing {len(video_paths)} videos"


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


def process_single_video(video_info, frame_interval=1, batch_size=32, input_queues=None):
    i, video_path = video_info
    video_id = video_path.stem
    queue_idx = i % len(input_queues)
    video_processor(str(video_path), video_id, input_queues[queue_idx],
                    frame_interval=frame_interval, batch_size=batch_size)
    return f"Processed {video_id}"


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
    # print(f"video_files: {video_files}")

    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    # for i, video_path in enumerate(video_files):
    #     video_id = video_path.stem
    #     print(f"video id: {video_id}")

    print(f"Found {len(video_files)} video files")

    # Initialize queues
    input_queues = [Queue() for _ in range(num_workers)]
    # input_queues = [Queue() for _ in range(len(gpu_ids) or 1)]
    output_queue = Queue()

    # Start inference workers (one per GPU, or one on CPU if no GPUs)
    inference_processes = []

    for i in range(num_workers):
        # TODO multi-process share same GPU?
        p = Process(target=inference_worker,
                    args=(model_path, input_queues[i], output_queue))
        p.start()
        inference_processes.append(p)
    print("inference processes started")
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
    print("collecting process started")

    num_video_workers = num_workers
    #Method 1:
    # Divide videos among workers instead of creating a process per video
    video_batches = []
    bs = len(video_files) // num_video_workers
    for i in range(num_video_workers):
        start_idx = i * bs
        end_idx = start_idx + bs if i < num_video_workers - 1 else len(video_files)
        video_batches.append(video_files[start_idx:end_idx])
    print(f"video_batches size: {len(video_batches)}")

    print("video processing processes started")
    # Start video processing workers - one process per batch of videos
    processing_processes = []
    for i, video_batch in enumerate(video_batches):
        # Each worker processes a batch of videos and sends frames to a specific queue
        queue_idx = i % len(input_queues)
        p = Process(target=video_processor_worker,
                    args=(i, video_batch, input_queues[queue_idx], frame_interval, batch_size))
        p.start()
        processing_processes.append(p)

    # Wait for video processing to complete
    for p in processing_processes:
        p.join()
    print("All video processing workers have completed")

    # #TODO Method 2: RuntimeError: Queue objects should only be shared between processes through inheritance
    # from functools import partial
    # process_func = partial(process_single_video,
    #                        frame_interval=frame_interval,
    #                        batch_size=batch_size,
    #                        input_queues=input_queues)
    #
    # with mp.Pool(processes=num_video_workers) as pool:
    #     video_infos = list(enumerate(video_files))
    #     results = pool.map(process_func, video_infos)

    # #TODO Method 3: Start video processing workers
    # processing_processes = []
    # for i, video_path in enumerate(video_files):
    #     video_id = video_path.stem
    #     # Distribute videos across available inference workers
    #     queue_idx = i % len(input_queues)
    #     # TODO ❓Potential problem? Each process consumes system resources (memory, file descriptors, process IDs).
    #     #  If too many processes are created simultaneously, it could exhaust system resources.
    #     p = Process(target=video_processor,
    #                 args=(str(video_path), video_id, input_queues[queue_idx]),
    #                 kwargs={'frame_interval': frame_interval, 'batch_size': batch_size})
    #     p.start()
    #     processing_processes.append(p)
    # # Wait for video processing to complete
    # for p in processing_processes:
    #     p.join()

    # Signal inference workers to terminate
    for q in input_queues:
        q.put(None)

    # Wait for inference workers to complete
    for p in inference_processes:
        p.join()

    # Wait for result collector to complete
    collector_process.join()

    print("All processing complete!")


def download_face_models():
    """Download necessary OpenCV face detection models if they don't exist"""
    import urllib.request
    import os

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Download face detection model
    face_detect_model = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
    face_detect_config = os.path.join(models_dir, "opencv_face_detector.pbtxt")

    if not os.path.exists(face_detect_model):
        print("Downloading face detection model...")
        urllib.request.urlretrieve(
            "https://github.com/ac005sheekar/Gender-Age-Detection-with-OpenCV-and-Keras/tree/master/opencv_face_detector_uint8.pb",
            face_detect_model
        )

    if not os.path.exists(face_detect_config):
        print("Downloading face detection model config...")
        urllib.request.urlretrieve(
            "https://github.com/ac005sheekar/Gender-Age-Detection-with-OpenCV-and-Keras/tree/master/opencv_face_detector.pbtxt",
            face_detect_config
        )

    print("Models downloaded successfully!")


if __name__ == "__main__":
    # download_face_models()

    import argparse

    parser = argparse.ArgumentParser(description="Extract 3DMM parameters from videos")
    parser.add_argument("--video_dir", type=str,
                        default="/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/face_cropped",
                        help="Directory containing video files")
    parser.add_argument("--model_path", type=str,
                        default="./pretrained_models/SMIRK_em1.pt",
                        help="Path to the 3DMM model")
    parser.add_argument("--result_dir", type=str,
                        default="/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/param",
                        help="Directory to save results to")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of video processing workers")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="GPU IDs to use")
    parser.add_argument("--frame_interval", type=int, default=1, help="Extract every nth frame")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of frames to process at once")

    args = parser.parse_args()

    main(args.video_dir, args.model_path, args.result_dir,
         args.num_workers, args.gpu_ids, args.frame_interval, args.batch_size)
