from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import Audio, DatasetDict, load_dataset
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


common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_11_0", "cs", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_11_0", "cs", split="test", use_auth_token=True)

print(common_voice)

"""Most ASR datasets only provide input audio samples (`audio`) and the 
corresponding transcribed text (`sentence`). Common Voice contains additional 
metadata information, such as `accent` and `locale`, which we can disregard for ASR.
Keeping the notebook as general as possible, we only consider the input audio and
transcribed text for fine-tuning, discarding the additional metadata information:
"""

common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

print(common_voice)

"""## Prepare Feature Extractor, Tokenizer and Data
"""


feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-small")

"""### Load WhisperTokenizer
"""


tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="Czech", task="transcribe")

"""### Combine To Create A WhisperProcessor
"""



"""### Prepare Data
"""

print(common_voice["train"][0])

"""Since 
our input audio is sampled at 48kHz, we need to _downsample_ it to 
16kHz prior to passing it to the Whisper feature extractor, 16kHz being the sampling rate expected by the Whisper model. 
"""


common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

"""Re-loading the first audio sample in the Common Voice dataset will resample 
it to the desired sampling rate:
"""

print(common_voice["train"][0])


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

common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)

common_voice.save_to_disk("../common-voice-11-hi")