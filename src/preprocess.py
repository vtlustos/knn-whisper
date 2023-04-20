from optparse import OptionParser
from datasets import Audio, DatasetDict, load_dataset, load_from_disk
from transformers import (WhisperFeatureExtractor,WhisperTokenizer)
from huggingface_hub.hf_api import HfFolder 

HfFolder.save_token('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')

def preprocess(dataset_name, dst_path, language=("cs", "Czech"), num_proc=16):
    pipeline_name = "openai/whisper-large-v2"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(pipeline_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        pipeline_name, language=language[1], task="transcribe")

    if dataset_name == "common_voice":
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", 
                                            language[0], split="train+validation",
                                            use_auth_token=True)
        common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0",
                                            language[0], split="test", 
                                            use_auth_token=True)
        common_voice = common_voice.remove_columns(
            ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

        # resample to 16kHz
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

        def prepare_dataset(batch):
            # load and resample audio data from 48 to 16kHz
            audio = batch["audio"]        
            # compute log-Mel input features from input audio array
            batch["input_features"] = feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            # encode target text to label ids
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
            return batch

        common_voice = common_voice.map(prepare_dataset, 
                                        remove_columns=common_voice.column_names["train"], 
                                        num_proc=num_proc)

        common_voice.save_to_disk(dst_path)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-d", "--dataset", dest="dataset_name",
                      help="Name of the dataset")
    parser.add_option("-o", "--output", dest="dst_path",
                    help="Output directory")
    parser.add_option("-l", "--language", dest="language",
                      help="Language to use (default: (cs,Czech))", default=("cs", "Czech"))
    parser.add_option("-n", "--num-proc", dest="num_proc",
                      help="Number of processes to use (default: 16)", default=16)

    (options, args) = parser.parse_args()

    preprocess(options.dataset_name, options.dst_path,
                options.language, options.num_proc)