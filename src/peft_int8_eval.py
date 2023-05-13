from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.distillation_trainer import DistillationTrainer
from utils.wer import WER
from optparse import OptionParser
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import gc
import evaluate

from datasets import load_dataset
from transformers import (WhisperForConditionalGeneration,WhisperProcessor)
from peft import (PeftModel, PeftConfig)

from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token("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") # token jkot

# for default dir paths
def eval(batch_size, 
          cache_dir="~/.cache/huggingface/datasets",
          model_name="vtlustos/whisper-large-v2_lora_int8"):

    # setup data pipeline
    pipeline_name = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(
        pipeline_name, language="czech", 
        task="transcribe"
    )

    # setup dataset    
    dataset_test_split = load_dataset("jkot/merged_preprocessed_parliament_commonvoice",
                                      split="test",
                                      cache_dir=cache_dir)
    print("Test dataset:", dataset_test_split)
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    peft_config = PeftConfig.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, model_name)
           
    eval_dataloader = DataLoader(dataset_test_split, batch_size=batch_size, collate_fn=data_collator)

    metric = evaluate.load("wer")

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        del generated_tokens, labels, batch
        gc.collect()
    wer = 100 * metric.compute()
    print(f"{wer=}")

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Batch size.", default=16)  
    parser.add_option("-c", "--cache-dir", dest="cache_dir",
                      default="~/.cache/huggingface/datasets")
    parser.add_option("-s", "--student-model-name", 
                        dest="student_model_name",
                        default="openai/whisper-small")
      
    (options, args) = parser.parse_args()

    print("Training with options: ", options)

    eval( 
        int(options.batch_size),
        options.cache_dir,
        options.student_model_name
    )