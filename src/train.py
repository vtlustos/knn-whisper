from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.distillation_trainer import DistillationTrainer
from utils.wer import WER
from optparse import OptionParser
from datasets import load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                          WhisperForConditionalGeneration, 
                          WhisperProcessor, TrainerCallback)
#from peft import prepare_model_for_training
from peft import LoraConfig, LoraConfig, get_peft_model
from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token("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")

# for default dir paths
def train(out_dir, 
          batch_size, 
          trainer,
          cache_dir="~/.cache/huggingface/datasets",
          efficient_tunning=False):

    # setup data pipeline
    pipeline_name = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(
        pipeline_name, language="czech", 
        task="transcribe"
    )

    # setup dataset
    dataset_train_split = load_dataset("jkot/dataset_merged_preprocesssed_v2", 
                                       split="train",
                                       cache_dir=cache_dir)
    print("Train dataset:", dataset_train_split)
    dataset_test_split = load_dataset("jkot/dataset_merged_preprocesssed_v2",
                                      split="test",
                                      cache_dir=cache_dir)
    print("Test dataset:", dataset_test_split)
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # initialize student and teacher models
    student_name = "openai/whisper-small" if trainer == "distill" else "openai/whisper-large-v2"
    student_model = WhisperForConditionalGeneration \
        .from_pretrained(student_name)
    student_model.config.forced_decoder_ids = processor \
        .get_decoder_prompt_ids(language="czech", task="transcribe")
    student_model.config.suppress_tokens = []
    
    # setup PEFT-LORA if required
    if(efficient_tunning):
        #student_model = prepare_model_for_training(student_model, output_embedding_layer_name="proj_out")
        lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], 
                                 lora_dropout=0.05, bias="none")
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()

    if trainer == "distill":
        teacher_model = WhisperForConditionalGeneration \
            .from_pretrained("openai/whisper-large-v2")
        teacher_model.config.forced_decoder_ids = processor \
            .get_decoder_prompt_ids(language="czech", task="transcribe")
        teacher_model.config.suppress_tokens = []
        teacher_model.to('cuda:0').half()

    # setup training process
    if efficient_tunning:
        training_args = Seq2SeqTrainingArguments(
            # paths
            output_dir=out_dir,
            push_to_hub=True,
            push_to_hub_model_id="whisper_large_v2_lora",
            push_to_hub_token="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",
            # model
            fp16=True,
            predict_with_generate=True,
            # batch
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            # learning rate
            learning_rate=1e-3,
            warmup_steps=500,
            max_steps=5000,
            generation_max_length=225,
            # feedback
            gradient_checkpointing=True,
            report_to=["tensorboard"],
            metric_for_best_model="wer",
            load_best_model_at_end=True,
            logging_first_step=True,
            logging_steps=5,
            save_steps=1000,
            eval_steps=1000,
            evaluation_strategy="steps",
            # PEFT-LORA specific
            remove_unused_columns=False,
            label_names=["labels"]
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            # paths
            output_dir=out_dir,
            push_to_hub=True,
            push_to_hub_model_id="whisper_large_v2",
            push_to_hub_token="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",
            # model
            fp16=True,
            predict_with_generate=True,
            # batch
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            # learning rate
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=5000,
            generation_max_length=225,
            # feedback
            gradient_checkpointing=True,
            report_to=["tensorboard"],
            logging_first_step=True,
            metric_for_best_model="wer",
            load_best_model_at_end=True,
            logging_steps=5,
            save_steps=1000,
            eval_steps=1000,
            evaluation_strategy="steps"
        )

    wer = WER(tokenizer=processor.tokenizer)
    if trainer == "seq2seq":
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=student_model,
            train_dataset=dataset_train_split,
            eval_dataset=dataset_test_split,
            data_collator=data_collator,
            compute_metrics=wer,
            tokenizer=processor.feature_extractor,
        )
    elif trainer == "distill":
        trainer = DistillationTrainer(
            config=training_args,
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataset=dataset_train_split,
            eval_dataset=dataset_test_split,
            tokenizer=processor.feature_extractor,
            data_collator=data_collator,
            compute_metrics=wer,
            temperature=2.0,
            supervised=False
        )

    # evaluate model before the first step
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True
    trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()
    trainer.push_to_hub()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-o", "--out-dir", dest="out_dir",
                        help="Path to the output directory.")
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Batch size.", default=16)
    parser.add_option("-t", "--trainer", dest="trainer",
                      help="Trainer (either seq2seq or distill)", default="seq2seq")
    parser.add_option("-c", "--cache-dir", dest="cache_dir",
                      default="~/.cache/huggingface/datasets")
    parser.add_option("-e", "--efficient-tunning", 
                      action="store_true",
                      dest="efficient_tunning")
  
    (options, args) = parser.parse_args()

    print("Training with options: ", options)

    train( 
        options.out_dir, 
        int(options.batch_size),
        options.trainer, 
        options.cache_dir,
        options.efficient_tunning
    )