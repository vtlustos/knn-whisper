# KNN WHISPER

This projects implements training, evaluation and dataset processing for our KNN speech recognition project. 

## Models
All of the trained models are available on vtlustos´s huggingface profile.

https://huggingface.co/vtlustos/whisper-base
https://huggingface.co/vtlustos/whisper-small
https://huggingface.co/vtlustos/whisper-large-v2_lora_int8

## Requirements
Project uses huggingface and pytorch. You can install all of the requirements using
```
pip install -r requirements.txt
```
## Dataset processing
To download, process and tokenize a dataset for training or evaluation, use the src/preprocess script.

### Arguments
--dataset: Name of the dataset (parliament, voxpopuli, common_voice)\
--output: Output directory\
--language: Language to use (default: (cs,Czech))\
--num-proc: Number of processes to use (default: 16)\


## Training
First preprocess the data. Then run src/train.py. 
### Arguments
--out-dir -o: Path to the output directory\
--batch-size -b: Batch size\
--cache-dir -c: Huggingface cache directory\
--student-model-name -s: Name of the model\
--teacher-model-name -t: Name of the teacher model (when using knowledge distillation trainer)\
--lora -l: Whether to use LORA\
--int8 -i: Whether to use int8

### Example

```bash
python src/train.py -o /storage/brno12-cerit/home/xtlust05/whisper/small/ -c $SCRATCHDIR -b 64 -s openai/whisper-small
python src/train.py -o /storage/brno12-cerit/home/xtlust05/whisper/large_v2/ -c $SCRATCHDIR -b 16 -s openai/whisper-large-v2
```

## Evaluation
To run evaluation of a model on processed dataset, use the src/eval.py script.

### Arguments
--out-dir -o: Path to the output directory\
--batch-size -b: Batch size\
--cache-dir -c: Huggingface cache directory\
--student-model-name -s: Name of the model\
-dataset-path -d: Path to preprocessed dataset with eval split

## PEFT model evaluation
To run evaluation on the PEFT model, please use the src/peft_int8_eval.py script.

### Arguments
--batch-size -b: Batch size\
--cache-dir -c: Huggingface cache directory\
--student-model-name -s: Name of the model\

## MetaCentrum
All the computation was done at the MetaCentrum.