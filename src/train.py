from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from utils.distillation_trainer import DistillationTrainer
from utils.wer import WER
from optparse import OptionParser
from datasets import load_from_disk, concatenate_datasets
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, 
                          WhisperForConditionalGeneration, 
                          WhisperProcessor, TrainerCallback)
from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token("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")
# for default dir paths
login ="xkotou06"
def train(common_voice_ds, 
          parliament_ds, 
          voxpopuli_ds, out_dir, batch_size, trainer, language="czech"):

    # setup data pipeline
    pipeline_name = "openai/whisper-large-v2"
    processor = WhisperProcessor.from_pretrained(
        pipeline_name, language=language, task="transcribe")

    # setup dataset
    common_voice = load_from_disk(common_voice_ds)
    parliament = load_from_disk(parliament_ds)
    voxpopuli = load_from_disk(voxpopuli_ds)
    merged_dataset_train_split = concatenate_datasets([common_voice["train"],parliament["train"], voxpopuli["train"]])
    merged_dataset_test_split = concatenate_datasets([common_voice["test"],  parliament["test"], voxpopuli["test"]])
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
        logging_steps=5,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
        push_to_hub_model_id="whisper_large_finetuned",
        push_to_hub_token="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",
        logging_first_step=True
    )

    wer = WER(tokenizer=processor.tokenizer)
    if trainer == "seq2seq":
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=student_model,
            train_dataset=merged_dataset_train_split,
            eval_dataset=merged_dataset_test_split,
            data_collator=data_collator,
            compute_metrics=wer,
            tokenizer=processor.feature_extractor,
        )
    elif trainer == "distill":
        trainer = DistillationTrainer(
            config=training_args,
            student_model=student_model,
            teacher_model=teacher_model,
            train_dataset=merged_dataset_train_split,
            eval_dataset=merged_dataset_test_split,
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
    parser.add_option("-c", "--commonvoice-dir", dest="commonvoice_dir",
                      help="Path to the commonvoice dataset directory.", default=f"/storage/brno12-cerit/home/{login}/common-voice-preprocessed_LARGEv2")
    parser.add_option("-p", "--parliament-dir", dest="parliament_dir",
                      help="Path to the commonvoice dataset directory.", default=f"/storage/brno12-cerit/home/{login}/czech_parliament_hearings_preprocessed_LARGEv2")
    parser.add_option("-v", "--voxpopuli-dir", dest="voxpopuli_dir",
                      help="Path to the commonvoice dataset directory.", default=f"/storage/brno12-cerit/home/{login}/voxpopulics_processed_LARGEv2")
    parser.add_option("-o", "--out-dir", dest="out_dir",
                      help="Path to the output directory.", default=f"/storage/brno12-cerit/home/{login}/whisperLargeOut")
    parser.add_option("-b", "--batch-size", dest="batch_size",
                      help="Batch size.", default=16)
    parser.add_option("-t", "--trainer", dest="trainer",
                      help="Trainer (either seq2seq or distill)", default="seq2seq")
    parser.add_option("-l", "--language", dest="language",
                      help="Language to use (default: cs,Czech)", default="czech")
  
    (options, args) = parser.parse_args()

    train(options.commonvoice_dir, options.parliament_dir, options.voxpopuli_dir, options.out_dir, int(options.batch_size),
        options.trainer, options.language
    )