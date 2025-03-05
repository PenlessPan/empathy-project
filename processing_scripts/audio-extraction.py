import os
from pydub import AudioSegment
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_audio(processed_movies_dir, output_dir):
    """
    Extract audio from MP4 files and save as 16kHz WAV files
    """
    os.makedirs(output_dir, exist_ok=True)

    for session_num in range(101, 141):
        if session_num in [117, 118, 139]:
            continue

        session_dir = os.path.join(processed_movies_dir, str(session_num))
        if not os.path.exists(session_dir):
            continue

        output_session_dir = os.path.join(output_dir, str(session_num))
        os.makedirs(output_session_dir, exist_ok=True)

        for filename in os.listdir(session_dir):
            if not filename.endswith('.mp4'):
                continue

            if 'no_sound' in filename:
                logger.info(f"Skipping {filename} as it has no sound")
                continue

            input_path = os.path.join(session_dir, filename)
            output_path = os.path.join(output_session_dir,
                                       filename.replace('.mp4', '.wav'))

            try:
                logger.info(f"Processing {filename}")
                # Load audio from MP4
                audio = AudioSegment.from_file(input_path, format="mp4")

                # Convert to mono if needed
                if audio.channels > 1:
                    audio = audio.set_channels(1)

                # Set frame rate to 16kHz
                audio = audio.set_frame_rate(16000)

                # Export as WAV
                audio.export(output_path, format='wav')

                logger.info(f"Successfully extracted audio from {filename}")

            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue

    return "Audio extraction completed"


if __name__ == "__main__":
    processed_movies_dir = "processed_movies"  # From previous step
    output_dir = "audio"
    result = extract_audio(processed_movies_dir, output_dir)
    print(result)