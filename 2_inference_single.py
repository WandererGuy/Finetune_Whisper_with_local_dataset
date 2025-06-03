import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "checkpoint-openai-whisper-small-en-vi/checkpoint-5000"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

import os 
processodr_model_id = os.path.dirname(model_id)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

input_wav_folder = "input_wav"
if not os.path.exists(input_wav_folder):
    os.makedirs(input_wav_folder)

import uuid

if __name__ == "__main__":
    language = "vietnamese"
    task = "transcribe"
    original_audio = "/mnt/d/WORK/dan_toc_projects/en_vn_dataset/wav_files/287df187-9a09-40c3-a52d-fd6a50c275e0.wav"
    name = str(uuid.uuid4()) + ".wav"
    converted_audio_path = os.path.join(input_wav_folder, name)
    convert_audio_to_wav(original_audio, converted_audio_path)
    result = pipe(converted_audio_path,generate_kwargs={"task": task, "language": language})
    print (f"Result: {result['text']}")