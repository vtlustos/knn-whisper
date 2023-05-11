import torch
from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.distillation_trainer import DistillationTrainer
from utils.wer import WER
from optparse import OptionParser
from datasets import load_from_disk, load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                          WhisperForConditionalGeneration, 
                          WhisperProcessor, TrainerCallback)

from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token("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") # token jkot

# for default dir paths
def train(out_dir, 
          batch_size, 
          cache_dir="~/.cache/huggingface/datasets",
          student_model_name="openai/whisper-small"):

    # setup data pipeline
    pipeline_name = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(
        pipeline_name, language="czech", 
        task="transcribe"
    )

    # setup dataset
    dset = load_from_disk("/scratch.ssd/xvlasa15/job_15406032.meta-pbs.metacentrum.cz/common-voice-13-small")
    dataset_train_split = dset["train"]
    # dataset_train_split = load_dataset("jkot/merged_preprocessed_parliament_commonvoice", 
    #                                    split="train",
    #                                    cache_dir=cache_dir)
    print("Train dataset:", dataset_train_split)
    # dataset_test_split = load_dataset("jkot/merged_preprocessed_parliament_commonvoice",
    #                                   split="test",
    #                                   cache_dir=cache_dir)
    dataset_test_split = dset["test"]
    print("Test dataset:", dataset_test_split)
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # initialize student and teacher models
    student_model = WhisperForConditionalGeneration \
        .from_pretrained(student_model_name
        )
    student_model.config.forced_decoder_ids = processor \
        .get_decoder_prompt_ids(language="czech", task="transcribe")
    student_model.config.suppress_tokens = []
    print("Student model:", student_model)


    training_args = Seq2SeqTrainingArguments(
        # paths
        output_dir=out_dir,
        
        # model
        fp16=True,
        predict_with_generate=True,
        generation_max_length=225,
        
        # batch
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1 if batch_size >= 16 else 16 // batch_size,
       
        # learning rate
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=10000,
        
        # output
        metric_for_best_model="wer",
        greater_is_better=True,
        load_best_model_at_end=True,

        # feedback
        report_to=["tensorboard"],
        logging_first_step=True,        
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        save_strategy = "steps",
        evaluation_strategy="steps",
    )

    print(training_args)

    wer = WER(tokenizer=processor.tokenizer)

    # casual seq-to-seq training
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=student_model,
        eval_dataset=dataset_test_split,
        data_collator=data_collator,
        compute_metrics=wer,
        tokenizer=processor.feature_extractor,
    )


    results = trainer.evaluate()
    print(results)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-o", "--out-dir", dest="out_dir",
                        help="Path to the output directory.")
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Batch size.", default=16)  
    parser.add_option("-c", "--cache-dir", dest="cache_dir",
                      default="~/.cache/huggingface/datasets")
    parser.add_option("-s", "--student-model-name", 
                        dest="student_model_name",
                        default="openai/whisper-small")
  
    (options, args) = parser.parse_args()

    print("Training with options: ", options)

    train( 
        options.out_dir, 
        int(options.batch_size),
        options.cache_dir,
        options.student_model_name
    )