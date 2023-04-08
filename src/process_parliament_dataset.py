from datasets import Audio, DatasetDict, load_dataset
from deepmultilingualpunctuation import PunctuationModel
import re
from transformers import WhisperFeatureExtractor, WhisperTokenizer

parliament_ds = DatasetDict()
parliament_ds["train"] = load_dataset("jkot/czech_parliament_plenary_hearings", split="train")
parliament_ds["test"] = load_dataset("jkot/czech_parliament_plenary_hearings",  split="test")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Czech", task="transcribe")
print(parliament_ds["train"][0])


def capitalize_after_interpunction(text):
    # Split the text into sentences using regular expressions
    sentences = re.split(r'(?<=[.?!])\s+', text)

    # Capitalize the first letter of each sentence after an interpunction
    for i in range(len(sentences)):
        print(len(sentences[i]))
        if(len(sentences[i]) != 0):
            sentences[i] = sentences[i][0].upper() + sentences[i][1:]

    # Join the sentences back into a string
    capitalized_text = ' '.join(sentences)

    return capitalized_text

def rm_SIL_token(string):
    if string.startswith("SIL "):
        string = string.lstrip("SIL ")
    if string.endswith(" SIL"):
        string = string.rstrip(" SIL")
    return string

def process_punctuation_casing(batch):
    uppercased = batch["transcription"]
    removed_sil = rm_SIL_token(uppercased)
    punctuated = capitalize_after_interpunction(removed_sil.lower())


    audio = batch["audio"]
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(punctuated).input_ids

    return batch

parliament_ds = parliament_ds.map(process_punctuation_casing, num_proc=4) 
parliament_ds.save_to_disk("../czech_parliament_plenary_hearings")