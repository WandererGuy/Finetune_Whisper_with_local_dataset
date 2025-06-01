Finetune whisper with local dataset 
unlike many tutorial online which shows dataset loaded from huggingface, this repo shows another alternative to use dataset from your own computer to finetune and inference

# preparation enivironment
- you will need 2 environments , one for finetune , one for inference because there are many conflict in dependency
- notice that whisper is built on 3.9 python in author's repo
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
you need to prepare a train dataset and a validation dataset. 
which means, you should prepare a train_csv and val_csv like in samples in ./sample_dataset which i already prepared
(example: the dataset i prepared is vietnamese ASR)
notice #1: sentence in each line is covered in "" so that csv reading wont fail
notice #2: each audio under audio column in your csv must be a 16k Hz + mono channel + wav file because that is audio type which whisper finetune need, so make sure to convert your audio file before any finetuning
i already prepare a file show how to convert your audio in ./0_wav_convert.py
# finetune pretrained whisper 
in 1_finetune_whisper.py, replace train_csv_path and test_csv_path with your own path to train csv and validation csv 

# inference pretrained whisper
