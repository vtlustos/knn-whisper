from src.utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.utils.distillation_trainer import DistillationTrainer
from src.utils.wer import WER
from optparse import OptionParser
from datasets import load_from_disk
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                          WhisperForConditionalGeneration, 
                          WhisperProcessor, TrainerCallback)
from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token('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')

def train(data_dir, out_dir, batch_size, trainer, language="czech"):

    # setup data pipeline
    pipeline_name = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(
        pipeline_name, language=language, task="transcribe")

    # setup dataset
    common_voice = load_from_disk(data_dir)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # initialize student and teacher models
    student_name = "openai/whisper-small" if trainer == "distill" else "openai/whisper-large-v2"
    student_model = WhisperForConditionalGeneration \
        .from_pretrained(student_name)
    student_model.config.forced_decoder_ids = processor \
        .get_decoder_prompt_ids(language=language, task="transcribe")
    student_model.config.suppress_tokens = []

    if trainer == "distill":
        teacher_model = WhisperForConditionalGeneration \
            .from_pretrained("openai/whisper-large-v2")
        teacher_model.config.forced_decoder_ids = processor \
            .get_decoder_prompt_ids(language=language, task="transcribe")
        teacher_model.config.suppress_tokens = []
        teacher_model.to('cuda:0').half()

    # setup training process
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=batch_size,
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

    wer = WER(tokenizer=processor.tokenizer)
    if trainer == "seq2seq":
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=student_model,
            train_dataset=common_voice["train"],
            eval_dataset=common_voice["test"],
            data_collator=data_collator,
            compute_metrics=wer,
            tokenizer=processor.feature_extractor,
        )
    elif trainer == "distill":
        trainer = DistillationTrainer(
            config=training_args,
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataset=common_voice["train"],
            eval_dataset=common_voice["test"],
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

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-d", "--data-dir", dest="data_dir",
                      help="Path to the dataset directory.")
    parser.add_option("-o", "--out-dir", dest="out_dir",
                      help="Path to the output directory.")
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Batch size.", default=32)
    parser.add_option("-t", "--trainer", dest="trainer",
                      help="Trainer (either seq2seq or distill)", default=("distill"))
    parser.add_option("-l", "--language", dest="language",
                      help="Language to use (default: cs,Czech)", default="czech")
  
    (options, args) = parser.parse_args()

    train(options.data_dir, options.out_dir, options.batch_size,
        options.trainer, options.language
    )