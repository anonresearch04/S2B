from transformers import BertConfig, BertForSequenceClassification

def get_model(model_name, num_labels):
    config = BertConfig.from_pretrained(model_name, cache_dir="./hf_cache", local_files_only=True)
    config.num_labels = num_labels
    model = BertForSequenceClassification(config)
    return model