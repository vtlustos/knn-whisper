
import evaluate

class WER:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = evaluate.load("wer")

    def __call__(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}