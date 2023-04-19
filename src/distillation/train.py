from src.distillation.utils.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.distillation.utils.trainer import DistillationTrainer
from src.distillation.utils.wer import WER
from optparse import OptionParser
from datasets import load_from_disk
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                          WhisperForConditionalGeneration, 
                          WhisperProcessor, TrainerCallback)
from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token('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')

def train(dataset_path, train_dir_path, language=("cs", "Czech")):

    # setup data pipeline
    pipeline_name = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(
        pipeline_name, language=language[1], task="transcribe")

    # setup dataset
    common_voice = load_from_disk(dataset_path)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # initialize student and teacher models
    student_model = WhisperForConditionalGeneration \
        .from_pretrained("openai/whisper-small")
    teacher_model = WhisperForConditionalGeneration \
        .from_pretrained("openai/whisper-large-v2")
    student_model.config.forced_decoder_ids = processor \
        .get_decoder_prompt_ids(language=language[1].lower(), task="transcribe")
    student_model.config.suppress_tokens = []
    teacher_model.config.forced_decoder_ids = processor \
        .get_decoder_prompt_ids(language=language[1].lower(), task="transcribe")
    teacher_model.config.suppress_tokens = []

    # setup training process
    training_args = Seq2SeqTrainingArguments(
        output_dir=train_dir_path,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
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
        logging_first_step=True
    )

    if False:
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=student_model,
            train_dataset=common_voice["train"],
            eval_dataset=common_voice["test"],
            data_collator=data_collator,
            compute_metrics=WER(tokenizer=processor.tokenizer),
            tokenizer=processor.tokenizer,
        )
    
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        temperature=2.0,
        supervised=False,
        args=training_args,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        compute_metrics=WER(tokenizer=processor.tokenizer),
        data_collator=data_collator,
        tokenizer=processor.tokenizer
    )

    # evaluate model before the first step
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True
    trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-d", "--dataset", dest="dataset_path",
                      help="Path to the dataset")
    parser.add_option("-t", "--train-dir", dest="train_dir_path",
                      help="Path to the training directory")
    parser.add_option("-l", "--language", dest="language",
                      help="Language to use (default: cs,Czech)", default=("cs", "Czech"))
    (options, args) = parser.parse_args()

    train(options.dataset_path, options.train_dir_path, options.language)