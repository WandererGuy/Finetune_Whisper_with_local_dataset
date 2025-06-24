from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
from transformers import WhisperForConditionalGeneration

import logging

# 1. Configure the root logger
logging.basicConfig(
    level=logging.INFO,                                   # minimum level to emit
    format="%(asctime)s %(levelname)s %(message)s",       # how each line is formatted
    datefmt="%Y-%m-%d %H:%M:%S"                           # timestamp format
)

# 2. (Optionally) get a named logger
logger = logging.getLogger(__name__)

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import Seq2SeqTrainingArguments

PRETRAIN_MODEL = "openai/whisper-medium"
# PRETRAIN_MODEL = "erax-ai/EraX-WoW-Turbo-V1.1-CT2"
# PRETRAIN_MODEL = "vinai/PhoWhisper-small"

OUTPUT_DIR_NAME = "checkpoint-" + PRETRAIN_MODEL.replace("/", "-")
LANGUAGE_WHISPER = "English"
MODEL_GENERATION_CONFIG_LANGUAGE = "english"
MODEL_TASK = "translate"

MAX_STEPS = 30000
 

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(
        self, 
        processor: Any,               # speech processor (feature_extractor + tokenizer)
        decoder_start_token_id: int   # thường là tokenizer.pad_token_id hoặc decoder_start_token_id
    ):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

import evaluate

sacrebleu = evaluate.load("sacrebleu")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # strip off extra whitespace
    pred_str = [s.strip() for s in pred_str]
    label_str = [s.strip() for s in label_str]

    # sacrebleu wants List[List[str]] for references
    references = [[ref] for ref in label_str]

    result = sacrebleu.compute(predictions=pred_str, references=references)
    # result["score"] is in percentage
    return {"bleu": result["score"]}

from transformers import Seq2SeqTrainingArguments


def load_dataset_from_local_files(train_csv_path, test_csv_path):
    """
    suppose you have 2 csv files train.csv and test.csv
    the format of the csv files should be like this:
    audio,sentence
    """
    from datasets import Dataset
    import pandas as pd
    from datasets import Audio

    ## we will load the both of the data here.
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    ## we will rename the columns as "audio", "sentence".
    train_df.columns = ["audio", "sentence"]
    test_df.columns = ["audio", "sentence"]

    ## convert the pandas dataframes to dataset 
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # convert the sample rate of every audio files using cast_column function
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    # train_dataset = train_dataset.map(prepare_dataset, num_proc=4)
    # test_dataset = test_dataset.map(prepare_dataset, num_proc=4)

    """
    train_dataset =
    Dataset({
        features: ['audio', 'sentence'],
        num_rows: 2
    })
    ----------------------------------------
    train_dataset[0] =
    {'audio': {'path': '/home/manh264/code_linux/Finetune_Whisper_translation/test_data/common_voice_de_17300032.mp3.wav', 
    'array': array([0.00756836, 0.00726318, 0.00759888, ..., 0.        , 0.        ,
           0.        ]), 
           'sampling_rate': 16000}, 
           'sentence': 'hello world'}
    """
    return train_dataset, test_dataset


if __name__ == "__main__":
        # train_csv_path = "/mnt/d/WORK/dan_toc_projects/Speech_dan_toc_crawl/train_linux.csv"
        # test_csv_path = "/mnt/d/WORK/dan_toc_projects/Speech_dan_toc_crawl/val_linux.csv"
        train_csv_path = "train_linux.csv"
        test_csv_path = "val_linux.csv"


        train_dataset, test_dataset = load_dataset_from_local_files(train_csv_path, test_csv_path)
        from datasets import Dataset
        # logger.info("done load dataset")
        logger.info("done load dataset")

        """
        The ASR pipeline can be de-composed into three components:

        A feature extractor which pre-processes the raw audio-inputs
        The model which performs the sequence-to-sequence mapping
        A tokenizer which post-processes the model outputs to text format

        """
        logger.info("start load whisper feature extractor")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(PRETRAIN_MODEL)
        logger.info("start load whisper tokenizer")
        tokenizer = WhisperTokenizer.from_pretrained(PRETRAIN_MODEL, language=LANGUAGE_WHISPER, task=MODEL_TASK)

        # input_str = common_voice["train"][0]["sentence"]
        # labels = tokenizer(input_str).input_ids
        # decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
        # decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

        # print(f"Input:                 {input_str}")
        # print(f"Decoded w/ special:    {decoded_with_special}")
        # print(f"Decoded w/out special: {decoded_str}")
        # print(f"Are equal:             {input_str == decoded_str}")


        def prepare_dataset(batch):
            # load and resample audio data from 48 to 16kHz
            audio = batch["audio"]

            # compute log-Mel input features from input audio array 
            batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

            # encode target text to label ids 
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
            return batch

        logger.info("start load whisper pretrained model + processor")
        processor = WhisperProcessor.from_pretrained(PRETRAIN_MODEL, language=LANGUAGE_WHISPER, task=MODEL_TASK)

        logger.info ("start prepare dataset")

        train_dataset = train_dataset.map(prepare_dataset, num_proc=1)
        test_dataset = test_dataset.map(prepare_dataset, num_proc=1)

        # max_mem = {0: "21GB"}

        model = WhisperForConditionalGeneration.from_pretrained(PRETRAIN_MODEL)
        model.generation_config.language = MODEL_GENERATION_CONFIG_LANGUAGE
        model.generation_config.task = MODEL_TASK

        model.generation_config.forced_decoder_ids = None

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )



        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./{OUTPUT_DIR_NAME}",  # change to a repo name of your choice
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=MAX_STEPS,
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            push_to_hub=False,
        )

        # training_args = Seq2SeqTrainingArguments(
        #     output_dir=f"./{OUTPUT_DIR_NAME}",  # change to a repo name of your choice
        #     per_device_train_batch_size=8,
        #     gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        #     learning_rate=1e-5,
        #     warmup_steps=500,
        #     max_steps=MAX_STEPS,
        #     gradient_checkpointing=True,
        #     fp16=True,
        #     eval_strategy="steps",
        #     per_device_eval_batch_size=8,
        #     predict_with_generate=True,
        #     generation_max_length=225,
        #     save_steps=1000,
        #     eval_steps=1000,
        #     logging_steps=25,
        #     report_to=["tensorboard"],
        #     load_best_model_at_end=True,
        #     metric_for_best_model="wer",
        #     greater_is_better=False,
        #     push_to_hub=False,
        # )




        # solve issue https://discuss.huggingface.co/t/tokenizer-not-created-when-training-whisper-small-model/61876/2
        # https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb#scrollTo=uOrRhDGtN5S4
        # We'll save the processor object once before starting training. Since the processor is not trainable, it won't change over the course of training:
        processor.save_pretrained(training_args.output_dir)

        from transformers import Seq2SeqTrainer

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )
        logger.info("start training")
        trainer.train()

