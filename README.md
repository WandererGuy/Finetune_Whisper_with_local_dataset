Finetune whisper with local dataset 
- unlike many tutorial online which shows dataset loaded from huggingface, this repo shows another alternative to use dataset from your own computer to finetune and inference
- u can use trained whisper for speech to text recogntion , or even speech to text translation (which i found as emergent property of whisper, eg: finetune on pairs of English audio and Vietnamese sentence, let task (in finetune and inference) still be transcribe not translate)
# preparation enivironment
- you will need 2 environments , one for finetune , one for inference because there are many conflict in dependency
- notice that whisper is built on 3.9 python like inn author's repo
## finetune env:
```
python==3.9 
pip install --upgrade pip
pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio
```
## infer env:
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
https://medium.com/@shridharpawar77/a-comprehensive-guide-for-custom-data-fine-tuning-with-the-whisper-model-60e4cbce736d
https://huggingface.co/blog/fine-tune-whisper

wer calculation: https://huggingface.co/learn/audio-course/en/chapter5/evaluation