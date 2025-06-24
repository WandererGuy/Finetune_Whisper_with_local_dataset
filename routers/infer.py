import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
device_en2vi = torch.device("cuda")
model_en2vi.to(device_en2vi)

def translate_en2vi(en_text: str) -> str:
    en_text = [en_text]
    input_ids = tokenizer_en2vi(en_text, padding=True, return_tensors="pt").to(device_en2vi)
    output_ids = model_en2vi.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    vi_texts = vi_texts[0]
    
    return vi_texts




from fastapi import FastAPI, HTTPException, Form, APIRouter, UploadFile, File
from routers.model import MyHTTPException, \
                        my_exception_handler, \
                        reply_bad_request, \
                        reply_server_error, \
                        reply_success
import os 
router = APIRouter()
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
static_folder = os.path.join(parent_dir, "static")
UPLOAD_DIR = os.path.join(static_folder, "upload_audio")
os.makedirs(UPLOAD_DIR, exist_ok=True)
INPUT_WAV_FOLDER = os.path.join(static_folder, "input_wav")
if not os.path.exists(INPUT_WAV_FOLDER):
    os.makedirs(INPUT_WAV_FOLDER)
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

model_id = "khm-en-medium-bleu/checkpoint-18000"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

import os 
processodr_model_id = os.path.dirname(model_id)
processor = AutoProcessor.from_pretrained(processodr_model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


import uuid 

@router.post("/whisper-translation-khmer/")
async def whisper_inference(file: UploadFile = File(...)):
    # Validate that it’s actually an audio file
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files are allowed")
    try:
        # Read the file into memory (beware large files!)
        contents = await file.read()

        # You could process the bytes here (e.g. transcribe, analyze, re-encode, etc.)
        extension = file.filename.split(".")[-1]
        save_name = str(uuid.uuid4()) + f".{extension}"
        save_path = os.path.join(UPLOAD_DIR, save_name)
        with open(save_path, "wb") as f:
            f.write(contents)


        language = "english"
        task = "translate"
        original_audio = save_path
        name = str(uuid.uuid4()) + ".wav"
        converted_audio_path = os.path.join(INPUT_WAV_FOLDER, name)
        convert_audio_to_wav(original_audio, converted_audio_path)
        result = pipe(converted_audio_path,generate_kwargs={"task": task, "language": language})
        en_text = result['text']
        text = translate_en2vi(en_text.strip())
        return reply_success("Success", {"english_text": en_text, "vietnamese_text": text})
    except Exception as e:
        return reply_server_error(e)