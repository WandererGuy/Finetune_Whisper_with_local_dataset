Finetune whisper with local dataset 
- unlike many tutorial online which shows dataset loaded from huggingface, this repo shows another alternative to use dataset from your own computer to finetune and inference
- u can use trained whisper for speech to text recogntion , or even speech to text translation (which i found as emergent property of whisper, eg: finetune on pairs of English audio and Vietnamese sentence, let task (in finetune and inference) still be transcribe not translate)
# preparation enivironment
- you will need 1 environments for both finetune and inference 
- in case inference cant be ran you can install a new env only for inference with instruction below 
- notice: in my env, whisper is built on 3.9 python like in Whisper's original repo 
## finetune + infer env:
linux
```
python==3.9 
pip install --upgrade pip
pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio
```
window
linux
```
python==3.9 
pip install --upgrade pip
pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio
pip install aiohttp==3.8.3q1
```
also, on window, you need to install pytorch from https://pytorch.org/get-started/locally/ or https://pytorch.org/get-started/previous-versions/ compatible for your cuda version


## infer env (Optional for inference only, if above env fail)
```
python==3.9
pip install -U openai-whisper
```

# prepare local dataset 
## prepare local dataset 
- you need to prepare a train dataset and a validation dataset. <br>
which means, you should prepare a train_csv and val_csv like in samples in ./sample_dataset which i already prepared <br>
(example: the dataset i prepared is vietnamese ASR) <br>
- notice #1: sentence in each line is covered in "" so that csv reading wont fail
- notice #2: each audio under audio column in your csv must be a 16k Hz + mono channel + wav file . <br>
Because that is audio type which whisper finetune need, so make sure to convert your audio file before any finetuning <br>
i already prepare a file show how to convert your audio in ./0_wav_convert.py
# finetune pretrained whisper 
- in 1_finetune_whisper.py, replace train_csv_path and test_csv_path with your own path to train csv and validation csv, which you prepared in previous step
- you need to change some values for whisper to work to your case <br>
In script ./1_finetune_whisper.py, you need to<br>
fix MODEL_GENERATION_CONFIG_LANGUAGE to your wanted finetune language (tag name must be available by whisper)<br>
fix LANGUAGE_WHISPER to your wanted finetune language (tag name must be available by whisper)<br>
u need a pretrained whisper model from huggingface to finetune, after select one, please fix PRETRAIN_MODEL
- run
```
python 1_finetune_whisper.py
```
- the checkpoint and logs will be saved in OUTPUT_DIR_NAME, specified in script ./1_finetune_whisper.py
# inference pretrained whisper
change variable : laguage , original_audio in python 2_inference.py to your preference
run 
```
python 2_inference_single.py
```


# other tutorial 
- https://medium.com/@shridharpawar77/a-comprehensive-guide-for-custom-data-fine-tuning-with-the-whisper-model-60e4cbce736d
- https://huggingface.co/blog/fine-tune-whisper
- wer calculation: https://huggingface.co/learn/audio-course/en/chapter5/evaluation