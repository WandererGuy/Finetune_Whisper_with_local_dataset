import subprocess
def convert_audio_to_wav(input_audio: str, output_wav: str):
    """
    Converts an MP3 file to a WAV file (16 kHz, mono, signed 16-bit PCM)
    using ffmpeg via subprocess.
    """
    cmd = [
        "ffmpeg",
        "-i", input_audio,
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        output_wav
    ]
    try:
        # Run and wait for completion; will raise CalledProcessError on non-zero exit
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Conversion succeeded: {output_wav}")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error converting {input_audio} to {output_wav}") from e

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP3 audio to WAV format.")
    parser.add_argument("input_audio", type=str, help="Path to the input MP3 audio file.")
    parser.add_argument("output_wav", type=str, help="Path to save the output WAV file.")
    args = parser.parse_args()
    input_audio = args.input_audio
    output_wav = args.output_wav
    # Example usage:
    try:
        convert_audio_to_wav(input_audio, output_wav)
    except Exception as e:
        print(f"Failed to convert audio: {e}")