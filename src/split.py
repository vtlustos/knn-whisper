from optparse import OptionParser
from datasets import load_dataset, concatenate_datasets

from huggingface_hub.hf_api import HfFolder 
HfFolder.save_token("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")

# for default dir paths
def split_ds(out_dir,          
          cache_dir="~/.cache/huggingface/datasets"):

    # load datasets
    dataset_train_split = load_dataset("jkot/dataset_merged_preprocessed", 
                                       split="train",
                                       cache_dir=cache_dir)
    dataset_test_split = load_dataset("jkot/dataset_merged_preprocessed",
                                      split="test",
                                      cache_dir=cache_dir)
    
    # merge and split datasets
    dataset_merged = concatenate_datasets([dataset_train_split, dataset_test_split])
    dataset_merged = dataset_merged.train_test_split(test_size=0.2, seed=42)

    dataset_merged.save_to_disk(out_dir)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-o", "--out-dir", dest="out_dir",
                        help="Path to the output directory.")
    parser.add_option("-c", "--cache-dir", dest="cache_dir",
                      default="~/.cache/huggingface/datasets")
  
    (options, args) = parser.parse_args()

    split_ds( 
        options.out_dir, 
        options.cache_dir
    )