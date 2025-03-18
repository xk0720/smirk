import os
import cv2
import torch
import numpy as np
import torch.multiprocessing as mp
from numpy import ndarray
from torch import Tensor
from torch.multiprocessing import Queue, Process
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from typing import List, Tuple
import traceback
from skimage.transform import estimate_transform, warp
from tqdm import tqdm
import detectors
from src.smirk_encoder import SmirkEncoder
from src import smirk_encoder
# import mediapipe
# from utils.mediapipe_utils import run_mediapipe


class FrameExtractor:
    """Extracts frames from videos and prepares them for the 3DMM model."""

    def __init__(self, frame_interval: int = 1, target_size: Tuple[int, int] = (224, 224),
                 detector=None, scale: float = 1.25):
        """
        Args:
            frame_interval: Extract every nth frame
            target_size: Size to resize frames to (height, width)
        """
        self.frame_interval = frame_interval
        # self.input_size = 512
        self.target_size = target_size
        self.scale = scale
        self.face_detector = detector

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type == 'kpt68':
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        elif type == 'bbox':
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.1])
            # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def crop_face(self, frame, landmarks, scale: float = 1.0, image_size: Tuple[int, int] = (224, 224)):
        # print("cropping face ...")
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

        size = int(old_size * scale)
        # print(f"old_size: {old_size}, size: {size}")

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_size[0] - 1], [image_size[1] - 1, 0]])
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
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"video_path: {video_path}, num_frames: {num_frames}, fps: {fps}")

        frames = []
        frame_count = 0

        pbar = tqdm(total=num_frames, desc="Processing Frames",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # #Method 1: =============================
            image = frame
            h, w, _ = image.shape
            bbox, bbox_type = self.face_detector.run(image)
            if len(bbox) < 4:
                print('no face detected! run original frame')
                left = 0;
                right = h - 1;
                top = 0;
                bottom = w - 1
            else:
                left = bbox[0];
                right = bbox[2];
                top = bbox[1];
                bottom = bbox[3];

            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)

            size = int(old_size * self.scale)
            src_pts = np.array(
                [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                 [center[0] + size / 2, center[1] - size / 2]])

            DST_PTS = np.array([[0, 0], [0, self.target_size[0] - 1], [self.target_size[1] - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            # image = image / 255.
            # dst_image = warp(image, tform.inverse, output_shape=(self.target_size[0], self.target_size[1]))
            # dst_image = dst_image.transpose(2, 0, 1)
            # cropped_image = torch.tensor(dst_image).float()/255.

            cropped_image = warp(image, tform.inverse,
                                 output_shape=(self.target_size[0], self.target_size[1]),
                                 preserve_range=True).astype(np.uint8)

            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = cv2.resize(cropped_image, (self.target_size[0], self.target_size[1]))

            # TODO debug: save cropped image for checking
            cv2.imwrite(f"cropped_image_{frame_count}.jpg", cropped_image)
            5/0

            cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).float() / 255.0
            # ========================================

            # #Method 2: =============================
            # kpt_mediapipe = run_mediapipe(frame)
            # ========================================

            # #Method 3: =============================
            # image = frame
            # image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=image_numpy)
            # detection_result = self.detector.detect(image)
            #
            # if len(detection_result.face_landmarks) == 0:
            #     print(f"No face detected at frame {frame_count}")
            #     # return None
            #
            # face_landmarks = detection_result.face_landmarks[0]
            # face_landmarks_numpy = np.zeros((478, 3))
            #
            # for i, landmark in enumerate(face_landmarks):
            #     face_landmarks_numpy[i] = [landmark.x * image.width, landmark.y * image.height, landmark.z]
            # kpt_mediapipe = face_landmarks_numpy
            # print("run mediapipe finished ...")

            # if kpt_mediapipe is None:
            #     print('Could not find landmarks for the image using mediapipe and cannot crop the face.')
            # # exit()
            #
            # kpt_mediapipe = kpt_mediapipe[..., :2]
            # tform = self.crop_face(frame, kpt_mediapipe, scale=1.2, image_size=self.target_size)
            # print("tform finished ...")
            #
            # cropped_image = warp(frame, tform.inverse, output_shape=self.target_size, preserve_range=True).astype(
            #     np.uint8)
            # # cropped_kpt_mediapipe = np.dot(tform.params,
            # #                                np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T).T
            # # cropped_kpt_mediapipe = cropped_kpt_mediapipe[:, :2]
            #
            # # Convert from BGR to RGB
            # cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            # cropped_image = cv2.resize(cropped_image, self.target_size)
            # cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).float() / 255.0
            # # [3, 224, 224]
            #
            # Resize frame
            # frame = cv2.resize(frame, self.target_size[::-1])  # cv2 expects (width, height)
            # ========================================

            frames.append(cropped_image)
            frame_count += 1

            pbar.update(1)
            # print(f"frame_count: {frame_count} / {num_frames}")

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


def inference_worker(model_path: str, input_queue: Queue, output_queue: Queue, gpu_id: int = 0,
                     result_dir: str = "/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/param"):
    """Worker process that runs the 3DMM model on GPU.

    Args:
        model_path: Path to the model file
        input_queue: Queue for receiving frames
        output_queue: Queue for sending results
        gpu_id: GPU device ID to use
        result_dir: Directory to save results
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

            # #Method 1: =============================
            video_id, temp_file = data
            frames_batch = torch.load(temp_file)
            # print(f"frames_batch size: {frames_batch.size()}")
            os.remove(temp_file)
            # print(f"got temp file: {temp_file} for video_id: {video_id}")
            # =============================

            # #Method 2: =============================
            # video_id, frames_batch = data
            # =============================

            # Extract 3DMM parameters
            parameters = model.extract_parameters(frames_batch)

            # save
            output_path = os.path.join(result_dir, f"{video_id}.npy")
            np.save(output_path, parameters)
            print(f"Saved parameters for {video_id} to {output_path}")

            # output_queue.put((video_id,))
            # output_queue.put((video_id, parameters))

    except Exception as e:
        print(f"Error in inference worker: {e}")
        traceback.print_exc()
        # Signal that there was an error
        output_queue.put(("ERROR", str(e)))
    finally:
        # Signal that we're done
        output_queue.put(None)


def video_processor_worker(worker_id, video_paths, input_queue, frame_interval=1,
                           batch_size=32, target_size: Tuple[int, int] = (224, 224),
                           temp_file_dir: str = "/lustre/projects/Research_Project-T127204/xk219/projects/datasets"
                                                "/HDTF/temp_tensors"):
    """Process a subset of videos and send frames to the inference queue"""
    detector = detectors.FAN()  # default 'FAN'
    print(f"detector loaded for worker {worker_id}")

    extractor = FrameExtractor(frame_interval, target_size, detector)
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
                # print("putting batch in queue")

                # #Method 1: save tensor and put url into queue
                batch_number = i // batch_size
                batch_id = f"{video_id}_{batch_number:06d}"  # Zero-padding to 6 digits
                # batch_id = f"{video_id}_{i // batch_size}"

                temp_file = os.path.join(temp_file_dir, f"{batch_id}.pt")
                torch.save(frames_batch, temp_file)
                print(f"saved temp file: {temp_file} for batch_id: {batch_id}")
                input_queue.put((batch_id, temp_file))

                # #Method 2: put tensor into queue
                # input_queue.put((f"{video_id}_{i // batch_size}", frames_batch))

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            traceback.print_exc()

    # Signal completion
    return f"Worker {worker_id} completed processing {len(video_paths)} videos"


# def video_processor(video_path: str, video_id: str, input_queue: Queue,
#                     frame_interval: int = 1, target_size: Tuple[int, int] = (224, 224),
#                     batch_size: int = 32):
#     """Process a video and extract frames for 3DMM parameter extraction.
#
#     Args:
#         video_path: Path to the video file
#         video_id: Identifier for the video
#         input_queue: Queue for sending frames to the inference worker
#         frame_interval: Extract every nth frame
#         target_size: Size to resize frames to
#         batch_size: Number of frames to process at once
#     """
#     try:
#         # Create a frame extractor
#         extractor = FrameExtractor(frame_interval, target_size)
#
#         # Extract frames
#         frames, fps = extractor.extract_frames(video_path)
#
#         print(f"Extracted {len(frames)} frames from {video_path}")
#
#         # Process frames in batches
#         for i in range(0, len(frames), batch_size):
#             batch_frames = frames[i:i + batch_size]  # maintain the remainder
#
#             # Preprocess frames
#             frames_batch = extractor.preprocess_frames(batch_frames)
#
#             # Put batch in the input queue
#             input_queue.put((f"{video_id}_{i // batch_size}", frames_batch))
#
#     except Exception as e:
#         print(f"Error processing video {video_path}: {e}")
#         traceback.print_exc()


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

            # #Method 1: don't save at the moment
            # print(f"got result: {result}")

            # #Method 2: Save the parameters
            video_id, parameters = result
            output_path = os.path.join(result_dir, f"{video_id}.npy")
            np.save(output_path, **parameters)
            print(f"Saved parameters for {video_id} to {output_path}")

    except Exception as e:
        print(f"Error in result collector: {e}")
        traceback.print_exc()


# def process_single_video(video_info, frame_interval=1, batch_size=32, input_queues=None):
#     i, video_path = video_info
#     video_id = video_path.stem
#     queue_idx = i % len(input_queues)
#     video_processor(str(video_path), video_id, input_queues[queue_idx],
#                     frame_interval=frame_interval, batch_size=batch_size)
#     return f"Processed {video_id}"


def resume(model_path: str, result_dir: str, frame_interval: int = 1, batch_size: int = 32):
    video_files = [
        "/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/face_cropped/MitchDaniels0_1.mp4",
        "/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/face_cropped/ByronDorgan1.mp4",
    ]

    video_ids = ["MitchDaniels0_1", "ByronDorgan1"]

    target_size = (224, 224)
    detector = detectors.FAN()  # default 'FAN'
    extractor = FrameExtractor(frame_interval, target_size, detector)
    model = Model3DMM(model_path, device="cuda:0")

    for i, video_path in enumerate(video_files):
        video_id = video_ids[i]
        try:
            # Extract frames
            frames, fps = extractor.extract_frames(str(video_path))

            # Process frames in batches
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                frames_batch = extractor.preprocess_frames(batch_frames)

                parameters = model.extract_parameters(frames_batch)

                batch_number = i // batch_size
                batch_id = f"{video_id}_{batch_number:06d}"  # Zero-padding to 6 digits
                output_path = os.path.join(result_dir, f"{batch_id}.npy")
                np.save(output_path, parameters)
                print(f"Saved parameters for {batch_id} to {output_path}")

        except Exception as e:
            print(f"Error raised {video_path}: {e}")
            traceback.print_exc()


def main(video_dir: str, model_path: str, result_dir: str,
         num_workers: int = 8, gpu_ids: List[int] = [0],
         frame_interval: int = 1, batch_size: int = 32,
         temp_file_dir: str = "/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/temp_tensors"):
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
    gpu_id = 0
    for i in range(num_workers):
        # TODO multi-process share same GPU?
        p = Process(target=inference_worker,
                    args=(model_path, input_queues[i], output_queue, gpu_id, result_dir))
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
    # collector_process = Process(target=result_collector,
    #                             args=(output_queue, result_dir, len(inference_processes)))
    # collector_process.start()
    # print("collecting process started")

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
    target_size = (224, 224)
    for i, video_batch in enumerate(video_batches):
        # Each worker processes a batch of videos and sends frames to a specific queue
        queue_idx = i % len(input_queues)
        p = Process(target=video_processor_worker,
                    args=(i, video_batch, input_queues[queue_idx], frame_interval,
                          batch_size, target_size, temp_file_dir))
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
    # collector_process.join()

    print("All processing complete!")


def load_detector():
    print("start loading detector ...")
    try:
        # Try with default settings (might use GPU)
        base_options = python.BaseOptions(model_asset_path='assets/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        print("Detector initialized successfully")
        return detector
    except Exception as e:
        print(f"Error initializing detector with default settings: {e}")
        print("Falling back to CPU-only mode...")

        # Try again with CPU delegate
        base_options = python.BaseOptions(
            model_asset_path='assets/face_landmarker.task',
            delegate="CPU"
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        print("Detector initialized in CPU-only mode")
        return detector


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    import argparse

    parser = argparse.ArgumentParser(description="Extract 3DMM parameters from videos")
    parser.add_argument("--video_dir", type=str,
                        default="/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/face_cropped",
                        help="Directory containing video files")
    parser.add_argument("--temp_file_dir", type=str,
                        default="/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/temp_tensors")
    parser.add_argument("--model_path", type=str,
                        default="./pretrained_models/SMIRK_em1.pt",
                        help="Path to the 3DMM model")
    parser.add_argument("--result_dir", type=str,
                        default="/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/param",
                        help="Directory to save results to")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of video processing workers")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="GPU IDs to use")
    parser.add_argument("--frame_interval", type=int, default=1, help="Extract every nth frame")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of frames to process at once")

    args = parser.parse_args()

    # resume(model_path=args.model_path,
    #        result_dir="/lustre/projects/Research_Project-T127204/xk219/projects/ai_digital_humans_repo_summary/"
    #                   "develop/smirk/temp_param_save",
    #        batch_size=128)

    main(args.video_dir, args.model_path, args.result_dir,
         args.num_workers, args.gpu_ids, args.frame_interval, args.batch_size)
