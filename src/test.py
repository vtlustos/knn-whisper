from transformers import pipeline

generator = pipeline(model="openai/whisper-tiny")
print(classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"))
