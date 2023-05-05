import os
from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.distillation_trainer import DistillationTrainer
from utils.wer import WER
from optparse import OptionParser
from datasets import load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                          WhisperForConditionalGeneration, 
                          WhisperProcessor, TrainerCallback)
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, TaskType
#from peft import prepare_model_for_training
from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token("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")

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# for default dir paths
def train(out_dir, 
          batch_size, 
          cache_dir="~/.cache/huggingface/datasets",
          student_model_name="openai/whisper-small",
          teacher_model_name=None):

    # setup data pipeline
    pipeline_name = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(
        pipeline_name, language="czech", 
        task="transcribe"
    )

    # setup dataset
    dataset_train_split = load_dataset("jkot/merged_preprocessed_parliament_commonvoice", 
                                       split="train",
                                       cache_dir=cache_dir)
    print("Train dataset:", dataset_train_split)
    dataset_test_split = load_dataset("jkot/merged_preprocessed_parliament_commonvoice",
                                      split="test",
                                      cache_dir=cache_dir)
    print("Test dataset:", dataset_test_split)
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # initialize student and teacher models
    student_model = WhisperForConditionalGeneration \
        .from_pretrained(student_model_name)
    student_model.config.forced_decoder_ids = processor \
        .get_decoder_prompt_ids(language="czech", task="transcribe")
    student_model.config.suppress_tokens = []
    print("Student model:", student_model)

    config = LoraConfig(r=32, 
                        lora_alpha=64, 
                        task_type=TaskType.SEQ_2_SEQ_LM, 
                        target_modules=["q", "v"], 
                        lora_dropout=0.05
                        )
    student_model = get_peft_model(student_model, config)
    student_model.print_trainable_parameters()
    
    if teacher_model_name != None:
        teacher_model = WhisperForConditionalGeneration \
            .from_pretrained(teacher_model_name)
        teacher_model.config.forced_decoder_ids = processor \
            .get_decoder_prompt_ids(language="czech", task="transcribe")
        teacher_model.config.suppress_tokens = []
        teacher_model.to('cuda:0').half()

    training_args = Seq2SeqTrainingArguments(
        # paths
        output_dir=out_dir,
        push_to_hub=True,
        push_to_hub_model_id=student_model_name.split("/")[-1],
        push_to_hub_token="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",
        
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
        max_steps=5000,
        
        # output
        metric_for_best_model="wer",
        greater_is_better=True,
        load_best_model_at_end=True,

        # feedback
        report_to=["tensorboard"],
        logging_first_step=False,        
        logging_steps=5,
        save_steps=1000,
        eval_steps=1000,
        save_strategy = "steps",
        evaluation_strategy="steps",

        # lora
        # remove_unused_columns=False,
        # label_names=["labels"]
    )

    wer = WER(tokenizer=processor.tokenizer)
    if teacher_model_name == None:
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=student_model,
            train_dataset=dataset_train_split,
            eval_dataset=dataset_test_split,
            data_collator=data_collator,
            compute_metrics=wer,
            tokenizer=processor.feature_extractor,
        )
    else:
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
            supervised=True
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
    parser.add_option("-c", "--cache-dir", dest="cache_dir",
                      default="~/.cache/huggingface/datasets")
    parser.add_option("-s", "--student-model-name", dest="student_model_name",
                        default="openai/whisper-small")
    parser.add_option("-t", "--teacher-model-name", dest="teacher_model_name",
                        default=None)
  
    (options, args) = parser.parse_args()

    print("Training with options: ", options)

    train( 
        options.out_dir, 
        int(options.batch_size),
        options.cache_dir,
        options.student_model_name,
        options.teacher_model_name
    )