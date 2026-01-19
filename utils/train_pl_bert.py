import utils.preprocessing as preprocess_utils
import utils.datasets as datasets_utils
from transformers import BertConfig, BertForSequenceClassification, TrainingArguments, Trainer


def train(model, train_dataset, validation_dataset, savepath, batch_size, epochs):
    training_args = TrainingArguments(
        output_dir=savepath,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=savepath,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )
    trainer.train()
    return model