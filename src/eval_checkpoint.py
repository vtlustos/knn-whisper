

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio

# load model and processor from model checkpoint
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="czech", task="transcribe")

ds = load_dataset("mozilla-foundation/common_voice_13_0", "cs", split="validation", streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

dataset_iterator = iter(ds)
for i in range(20):
    input = next(dataset_iterator)
    input_features = processor(input["audio"]["array"], sampling_rate=input["audio"]["sampling_rate"], return_tensors="pt").input_features
    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    expected = input["sentence"]
    print(f"trn: {transcription}")
    print(f"exp: {expected}")