from datasets import Dataset, GeneratorBasedBuilder, Features
import os
import tarfile
import librosa
import datasets
import spacy
_LICENSE = "https://creativecommons.org/licenses/by/4.0/"
_HOMEPAGE = "https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3126"
_DATASET_URL = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3126/snemovna.tar.xz"

_DESCRIPTION = "Large corpus of Czech parliament plenary sessions, originaly released 2019-11-29 by Kratochvíl Jonáš, Polák Peter and Bojar Ondřej\
            The dataset consists of 444 hours of transcribed speech audio snippets 1 to 40 seconds long.\
            Original dataset transcriptions were converted to true case from uppercase using spacy library."

_CITATION = """\
 @misc{11234/1-3126,
 title = {Large Corpus of Czech Parliament Plenary Hearings},
 author = {Kratochv{\'{\i}}l, Jon{\'a}{\v s} and Pol{\'a}k, Peter and Bojar, Ond{\v r}ej},
 url = {http://hdl.handle.net/11234/1-3126},
 note = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
 copyright = {Creative Commons - Attribution 4.0 International ({CC} {BY} 4.0)},
 year = {2019} } """


class CzechParliamentPlenaryHearings(GeneratorBasedBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "audio": datasets.features.Audio(sampling_rate=16000),
                    "transcription": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE
        )
    
    def add_punctuation(text):
        doc = nlp(text)
        # Tokenize the text using the Transformers tokenizer
        inputs = tokenizer(text, return_tensors="pt")

        # Use the Transformers model to predict the punctuation for each token
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)

        # Combine the Spacy tokens with the predicted punctuation to form the final punctuated text
        punctuated_text = ""
        for i, token in enumerate(doc):
            punctuated_text += token.text
            if i < len(doc) - 1 and predictions[0][i+1] == 1:
                punctuated_text += " "

        return punctuated_text

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_DATASET_URL)
        data_dir = os.path.join(data_dir, 'ASR_DATA')
        splits = ("train", "dev", "test")

        split_names = {
            "train": datasets.Split.TRAIN,
            "dev": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }
        split_generators = []
        for split in splits:
            split_generators.append(
                datasets.SplitGenerator(
                    name=split_names.get(split, split),
                    gen_kwargs={'split': split, 'data_dir': data_dir}
                )
            )
        return split_generators

    def _generate_examples(self, split, data_dir):
        split_dir = os.path.join(data_dir, split)
        for folder_name in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder_name)
            if os.path.isdir(folder_path):
                for audio_file in os.listdir(folder_path):
                    if audio_file.endswith('.wav'):
                        audio_path = os.path.join(folder_path, audio_file)
                        if split == "dev":
                            transcription_path = os.path.join(folder_path, audio_file[:-4] + '.txt')     
                        else:
                            transcription_path = os.path.join(folder_path, audio_file + '.trn')
                        transcription = open(transcription_path).read().strip()

                        audio, sr = librosa.load(audio_path, sr=16000)
                        id = f"{folder_name}/{audio_file}"
                        yield id, {
                            'id': id,
                            'audio': {
                                'path': audio_path,
                                'bytes': audio.tobytes()
                            },
                            'transcription': transcription,
                        }
