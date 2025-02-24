import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from tqdm import tqdm
from src.smirk_encoder import SmirkEncoder
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
import argparse
import os
import src.utils.masking as masking_utils
from utils.mediapipe_utils import run_mediapipe
from datasets.base_dataset import create_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from multiprocessing import Manager


class VideoTracking(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_image_size = 224

        # load motion coefficients encoder
        self.smirk_encoder = SmirkEncoder().to(args.device)
        checkpoint = torch.load(args.checkpoint)
        checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if
                              'smirk_encoder' in k}  # checkpoint includes both smirk_encoder and smirk_generator

        self.smirk_encoder.load_state_dict(checkpoint_encoder)
        self.smirk_encoder.eval()
        self.smirk_encoder.share_memory()

        # instantiate FLAME model
        self.flame = FLAME().to(args.device)
        self.flame.share_memory()

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

    def extract(self, args):
        input_video_path, output_3dmm_path, shared_queue = args

        # create video file
        cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            print('Error opening video file')
            error_message = f"Video opening error happened: {input_video_path}"
            shared_queue.put(error_message)
            exit()

        # get the original frame rate of the video
        video_fps = cap.get(cv2.CAP_PROP_FPS)  # todo further processing based on fps?

        # video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        frame_ind = 0
        coeffs_list = []  # np.zeros(shape=(frame_count, 1, 58))

        while True:
            # loading frames
            ret, image = cap.read()

            # If the frame was not read successfully, end of the video is reached
            if not ret:
                break

            kpt_mediapipe = run_mediapipe(image)
            # no face detected
            if kpt_mediapipe is None:
                print(f"No face is detected in frame {frame_ind + 1}.")
                error_message = f"Face detection error happened: {input_video_path}"
                shared_queue.put(error_message)
                break

            # crop face if needed
            if args.crop:
                if (kpt_mediapipe is None):
                    print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
                    exit()

                kpt_mediapipe = kpt_mediapipe[..., :2]
                tform = self.crop_face(image, kpt_mediapipe, scale=1.4, image_size=self.input_image_size)

                cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(
                    np.uint8)

                cropped_kpt_mediapipe = np.dot(tform.params,
                                               np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T).T
                # cropped_kpt_mediapipe = cropped_kpt_mediapipe[:, :2]
            else:
                cropped_image = image
                # cropped_kpt_mediapipe = kpt_mediapipe

            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            cropped_image = cv2.resize(cropped_image, (224, 224))
            cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            cropped_image = cropped_image.to(args.device)

            with torch.no_grad():
                # pose_outputs = self.pose_encoder(img)
                # shape_outputs = self.shape_encoder(img)
                # expression_outputs = self.expression_encoder(img)

                outputs = self.smirk_encoder(cropped_image)
                expression = outputs['expression_params']
                jaw = outputs['jaw_params']
                pose = outputs['pose_params']
            coeffs_3dmm = torch.cat((expression, jaw, pose), dim=-1)  # dimensionality

            print(f"shape of coeffs_3dmm: {coeffs_3dmm.shape}")
            5/0

            # # todo load images
            # images = None

            coeffs_list.append(coeffs_3dmm)

            # save 3DMM at the moment
            coeffs_list.append(coeffs_3dmm.detach().cpu())

            frame_ind += 1
            if frame_ind >= frame_count:
                break

        if len(coeffs_list) < frame_count:
            print("Some frames have no faces detected.")
        else:
            print("All frames detected faces, saving npy file ...")
            all_coeffs = torch.stack(coeffs_list, dim=0).numpy()
            print("The shape of extracted 3DMM coefficients: ", all_coeffs.shape)

            try:
                # Save the array
                np.save(output_3dmm_path, all_coeffs)
                print(f"Successfully saved audio clip to {output_3dmm_path}")
            except Exception as e:
                error_type = type(e).__name__
                if isinstance(e, IOError):
                    print(f"IO Error: Unable to save file to {output_3dmm_path}")
                elif isinstance(e, ValueError):
                    print(f"Value Error: Error in saving NumPy array")
                else:
                    print(f"Unexpected error occurred while saving the file")

                print(f"Error details: {str(e)}")

                # record the trouble url
                error_message = f"Saving error happened: {input_video_path}"
                shared_queue.put(error_message)


def main(cfg):
    root_path = cfg.root  # csv file

    # list_path = pd.read_csv(os.path.join(root_path, split + '_{}.csv'.format(dataset)),
    #                         header=None, delimiter=',').drop(0)

    # len_processed_files = len(processed_files)
    # print("len_processed_files: ", len_processed_files)
    # participant_paths = participant_paths[len_processed_files:]
    # print("length of new participant_paths: ", len(participant_paths))
    # print("participant_paths: ", participant_paths)

    # TODO do something
    input_dir = ""
    output_dir = ""
    participant_paths = []

    args_list = []
    for path in participant_paths:
        # save_dir = '/'.join(os.path.join(output_dir, path).split('/')[:-1])
        # "/phd_data_all/UDIVA_clean/test/3D_FV_files/UDIVA/animal/FC1"
        # os.makedirs(save_dir, exist_ok=True)

        input_path = os.path.join(input_dir, path + '.mp4')
        output_path = os.path.join(output_dir, path + '.npy')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        args_list.append((input_path, output_path))
        # example:
        # ('/root/autodl-tmp/PhD_code_exp/phd_data_all/UDIVA_clean/test/Video_files/UDIVA/talk/188189/FC2/9.mp4',
        # '/root/autodl-tmp/PhD_code_exp/phd_data_all/UDIVA_clean/test/3D_FV_files/UDIVA/talk/188189/FC2/9.npy')

    if torch.cuda.device_count() > 0:
        # Adjust the device ordinal as needed
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    track_model = VideoTracking(cfg, device)

    with Manager() as manager:
        shared_queue = manager.Queue()

        args_list = [args + shared_queue for args in args_list]
        # for instance: [(input_path, output_path, shared_queue)]

    with mp.Pool(cfg.num_processing) as p:
        with tqdm(total=len(args_list), desc="extracting FLAME blendshapes from video files") as pbar:
            for path in p.imap_unordered(func=track_model.tracking,
                                         iterable=args_list,
                                         chunksize=cfg.chunksize):  # HYPEParameter setup
                pbar.update()
                print(f"Done processed video file: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/mead_90.png', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='trained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    parser.add_argument('--out_path', type=str, default='output',
                        help='Path to save the output (will be created if not exists)')
    parser.add_argument('--use_smirk_generator', action='store_true',
                        help='Use SMIRK neural image to image translator to reconstruct the image')
    parser.add_argument('--render_orig', action='store_true',
                        help='Present the result w.r.t. the original image/video size')

    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    main(cfg=parser.parse_args())
