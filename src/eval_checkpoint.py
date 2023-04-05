

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="czech", task="transcribe")

ds = load_dataset("mozilla-foundation/common_voice_13_0", "cs", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))


input = next(iter(ds))
input_speech = input["audio"]
input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

print(f"trn tokens: {transcription}")


transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
expected = input["sentence"]

print(f"trn: {transcription}")
print(f"exp: {expected}")