from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.distillation_trainer import DistillationTrainer
from utils.wer import WER
from optparse import OptionParser
from datasets import load_dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                          WhisperForConditionalGeneration, 
                          WhisperProcessor, TrainerCallback)
from peft import (prepare_model_for_int8_training, LoraConfig, 
                  PeftModel, LoraModel, LoraConfig, 
                  get_peft_model, TaskType)

from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token("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") # token jkot

# for default dir paths
def train(out_dir, 
          batch_size, 
          cache_dir="~/.cache/huggingface/datasets",
          student_model_name="openai/whisper-small",
          teacher_model_name=None,
          lora=False,
          int8=False):

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
    if int8:
        student_model = WhisperForConditionalGeneration \
            .from_pretrained(student_model_name, load_in_8bit=True, device_map="auto")
        student_model = prepare_model_for_int8_training(student_model)
    else:
        student_model = WhisperForConditionalGeneration \
            .from_pretrained(student_model_name)
        
    student_model.config.forced_decoder_ids = processor \
        .get_decoder_prompt_ids(language="czech", task="transcribe")
    student_model.config.suppress_tokens = []
    print("Student model:", student_model)

    if lora:
        config = LoraConfig(
            r=16, 
            lora_alpha=16, 
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        student_model = get_peft_model(student_model, config)
        student_model.print_trainable_parameters()
        print(student_model)
    
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
        hub_model_id =student_model_name.split("/")[-1],
        push_to_hub_token="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", # token vtlustos
        
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
        warmup_steps=50,
        max_steps=5000,

        # output
        #metric_for_best_model="wer",
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
    if lora:
        training_args.hub_model_id += "_lora"
        # training_args.gradient_checkpointing=False  # lora does not support gradient checkpointing
        training_args.learning_rate = 1e-4          # higher LR for lora
        training_args.save_strategy = "no"          # needed for PEFT
        training_args.remove_unused_columns=False   # needed for PEFT
        training_args.label_names=["labels"]        # needed for PEFT
    
    if int8:
        training_args.hub_model_id += "_int8"
        training_args.predict_with_generate = False

    print(training_args)

    wer = WER(tokenizer=processor.tokenizer)
    if teacher_model_name == None:
        # casual seq-to-seq training
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=student_model,
            train_dataset=dataset_train_split,
            eval_dataset=dataset_test_split,
            data_collator=data_collator,
            #compute_metrics=wer,
            tokenizer=processor.feature_extractor,
        )
    else:
        # knowledge distillation training
        training_args.hub_model_id += "_distilled"
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

    # save model
    trainer.model.save_pretrained(training_args.output_dir)                 # trained PEFT + LORA model
    trainer.model.base_model.save_pretrained(training_args.output_dir)      # base model
    processor.feature_extractor.save_pretrained(training_args.output_dir)   # tokenizer 
    trainer.push_to_hub()

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
    parser.add_option("-t", "--teacher-model-name", 
                        dest="teacher_model_name",
                        default=None)
    parser.add_option("-l", "--lora", dest="lora",
                        action="store_true",
                        default=False)
    parser.add_option("-i", "--int8", dest="int8",
                        action="store_true",
                        default=False)
  
    (options, args) = parser.parse_args()

    print("Training with options: ", options)

    train( 
        options.out_dir, 
        int(options.batch_size),
        options.cache_dir,
        options.student_model_name,
        options.teacher_model_name,
        options.lora,
        options.int8
    )