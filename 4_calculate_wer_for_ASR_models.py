from transformers.models.whisper.english_normalizer import BasicTextNormalizer

normalizer = BasicTextNormalizer()
from evaluate import load

wer_metric = load("wer")
a = "translated_output.csv"
sum_wer = 0
count = 0
import pandas as pd
df = pd.read_csv(a)
for idx, row in df.iterrows():

    reference = row["original_sentence"]
    prediction = row["translated_text"]
    normalized_prediction = normalizer(prediction)
    normalized_referece = normalizer(reference)
    wer = wer_metric.compute(
        references=[normalized_referece], predictions=[normalized_prediction]
    )
    count += 1
    sum_wer += wer
    print (f"WER: {wer * 100:.2f}%")

average_wer = sum_wer / count if count > 0 else 0
print(f"Average WER: {average_wer * 100:.2f}%")