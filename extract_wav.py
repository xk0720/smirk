import os
import subprocess
from tqdm import tqdm
import argparse


def extract_audio(input_video_path, output_wav_path):
    """
    Extract WAV audio from video file using FFmpeg

    Args:
        input_video_path: Path to the input video file
        output_wav_path: Path to save the output WAV file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if the input video exists
        if not os.path.exists(input_video_path):
            print(f'Error: Input video file does not exist: {input_video_path}')
            return False

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

        # FFmpeg command to extract audio: -y forces overwrite if file exists
        cmd = [
            'ffmpeg', '-y',  # Force overwrite existing files
            '-i', input_video_path,  # Input file
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian audio codec
            '-ar', '16000',  # Audio sampling rate: 16kHz
            '-ac', '1',  # Audio channels: 1 (mono)
            output_wav_path  # Output file
        ]

        # Run the FFmpeg command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if the command was successful
        if result.returncode != 0:
            print(f"Error extracting audio from {input_video_path}")
            print(f"FFmpeg error: {result.stderr.decode('utf-8')}")
            return False

        print(f"Successfully extracted audio to {output_wav_path}")
        return True

    except Exception as e:
        error_type = e.__class__.__name__
        print(f"Error extracting audio: {error_type}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract WAV audio from video files')
    parser.add_argument('--input_dir', type=str,
                        # default="/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF_dataset/HDTF",
                        default="/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/video_clipped",
                        help='Directory containing input video files')
    parser.add_argument('--output_dir', type=str,
                        default="/lustre/projects/Research_Project-T127204/xk219/projects/datasets/HDTF/wav",
                        help='Directory to save extracted audio files')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []

    for filename in os.listdir(args.input_dir):
        for ext in video_extensions:
            if filename.lower().endswith(ext):
                video_files.append(filename)
                break

    # Process each video file
    failed_files = []

    with tqdm(total=len(video_files), desc="Extracting audio from videos") as pbar:
        for video_file in video_files:
            input_path = os.path.join(args.input_dir, video_file)

            # Create output path with .wav extension
            base_name = os.path.splitext(video_file)[0]
            output_path = os.path.join(args.output_dir, f"{base_name}.wav")
            print(f"Processing {input_path} to {output_path}")
            5/0

            # Extract audio
            success = extract_audio(input_path, output_path)

            if not success:
                failed_files.append(input_path)

            pbar.update(1)

    # Report results
    print(f"Audio extraction completed. Processed {len(video_files)} files.")

    if failed_files:
        print(f"Failed to process {len(failed_files)} files.")

        # Save failed files list
        failed_log_path = "failed_audio_extractions.txt"
        with open(failed_log_path, "w", encoding="utf-8") as f:
            for file_path in failed_files:
                f.write(f"{file_path}\n")

        print(f"List of failed files saved to {failed_log_path}")


if __name__ == "__main__":
    main()
