from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import Audio, DatasetDict, load_dataset,load_from_disk
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperFeatureExtractor,
                          WhisperForConditionalGeneration, WhisperProcessor,
                          WhisperTokenizer)

# notebook_login()
from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token('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')


"""## Load Dataset

Using 🤗 Datasets, downloading and preparing data is extremely simple. 
We can download and prepare the Common Voice splits in just one line of code. 

First, ensure you have accepted the terms of use on the Hugging Face Hub: [mozilla-foundation/common_voice_11_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0). Once you have accepted the terms, you will have full access to the dataset and be able to download the data locally.

Since Hindi is very low-resource, we'll combine the `train` and `validation` 
splits to give approximately 8 hours of training data. We'll use the 4 hours 
of `test` data as our held-out test set:
"""


# common_voice = DatasetDict()

# common_voice["train"] = load_dataset(
#     "mozilla-foundation/common_voice_11_0", "cs", split="train+validation", use_auth_token=True)
# common_voice["test"] = load_dataset(
#     "mozilla-foundation/common_voice_11_0", "cs", split="test", use_auth_token=True)

# print(common_voice)

# """Most ASR datasets only provide input audio samples (`audio`) and the 
# corresponding transcribed text (`sentence`). Common Voice contains additional 
# metadata information, such as `accent` and `locale`, which we can disregard for ASR.
# Keeping the notebook as general as possible, we only consider the input audio and
# transcribed text for fine-tuning, discarding the additional metadata information:
# """

# common_voice = common_voice.remove_columns(
#     ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

# print(common_voice)

# """## Prepare Feature Extractor, Tokenizer and Data
# """


# feature_extractor = WhisperFeatureExtractor.from_pretrained(
#     "openai/whisper-small")

# """### Load WhisperTokenizer
# """


tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="Czech", task="transcribe")

"""### Combine To Create A WhisperProcessor
"""


processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="Czech", task="transcribe")

"""### Prepare Data
"""

# print(common_voice["train"][0])

"""Since 
our input audio is sampled at 48kHz, we need to _downsample_ it to 
16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model. 
"""


# common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

"""Re-loading the first audio sample in the Common Voice dataset will resample 
it to the desired sampling rate:
"""

# print(common_voice["train"][0])


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


"""We can apply the data preparation function to all of our training examples using dataset's `.map` method. The argument `num_proc` specifies how many CPU cores to use. Setting `num_proc` > 1 will enable multiprocessing. If the `.map` method hangs with multiprocessing, set `num_proc=1` and process the dataset sequentially."""
# common_voice = common_voice.map(
#     prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

common_voice = load_from_disk("../common-voice-13")

"""## Training and Evaluation
"""


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


"""Let's initialise the data collator we've just defined:"""

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

"""### Evaluation Metrics
"""


metric = evaluate.load("wer")

"""We then simply have to define a function that takes our model 
predictions and returns the WER metric. This function, called
`compute_metrics`, first replaces `-100` with the `pad_token_id`
in the `label_ids` (undoing the step we applied in the 
data collator to ignore padded tokens correctly in the loss).
It then decodes the predicted and label ids to strings. Finally,
it computes the WER between the predictions and reference labels:
"""


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


"""### Load a Pre-Trained Checkpoint
"""


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

"""Override generation arguments - no tokens are forced as decoder outputs (see [`forced_decoder_ids`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids)), no tokens are suppressed during generation (see [`suppress_tokens`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.suppress_tokens)):"""

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

"""### Define the Training Configuration

In the final step, we define all the parameters related to training. For more detail on the training arguments, refer to the Seq2SeqTrainingArguments [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).
"""


training_args = Seq2SeqTrainingArguments(
    output_dir="/storage/brno12-cerit/home/xvlasa15/WhisperSmallCommonVoice",  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1.25e-5,
    warmup_steps=500,
    max_steps=20000,
    gradient_checkpointing=True,
    fp16=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

"""**Note**: if one does not want to upload the model checkpoints to the Hub, 
set `push_to_hub=False`.
"""


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

"""We'll save the processor object once before starting training. Since the processor is not trainable, it won't change over the course of training:"""

processor.save_pretrained(training_args.output_dir)

"""### Training
"""

trainer.train()

# kwargs = {
#     "dataset_tags": "mozilla-foundation/common_voice_11_0",
#     "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
#     "dataset_args": "config: hi, split: test",
#     "language": "hi",
#     "model_name": "Whisper Small Hi - Sanchit Gandhi",  # a 'pretty' name for our model
#     "finetuned_from": "openai/whisper-small",
#     "tasks": "automatic-speech-recognition",
#     "tags": "hf-asr-leaderboard",
# }

"""The training results can now be uploaded to the Hub. To do so, execute the `push_to_hub` command and save the preprocessor object we created:"""

# trainer.push_to_hub(**kwargs)
