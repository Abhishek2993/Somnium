"""
train_distilbert.py — model (c, optional): fine-tune DistilBERT for
genre classification. Gated behind ENABLE_DISTILBERT (backend/.env,
default false) because fine-tuning a transformer on CPU (no GPU is
available in the dev environment this was built in) is slow even for
a small dataset -- this is written and correct but has NOT been run
end-to-end here. Enable the flag and install `transformers`+`torch`
to use it for real.
"""
import os

import paths  # noqa: F401

import config

RANDOM_SEED = 42
DEFAULT_MODEL_NAME = 'distilbert-base-uncased'


def is_enabled() -> bool:
    return config.ENABLE_DISTILBERT


def fit(texts: list[str], labels: list[str], seed: int = RANDOM_SEED,
        model_name: str = DEFAULT_MODEL_NAME, epochs: int = 3, output_dir: str | None = None) -> dict:
    """
    Fine-tunes DistilBERT for sequence classification. Returns
    {'model', 'tokenizer', 'label_list'}. Raises RuntimeError if
    ENABLE_DISTILBERT is false, so this never runs by accident.
    """
    if not is_enabled():
        raise RuntimeError(
            'ENABLE_DISTILBERT is false. Set it to true in backend/.env (and install '
            '`transformers`+`torch`) to fine-tune DistilBERT.'
        )

    import torch
    from torch.utils.data import Dataset
    from transformers import (
        DistilBertForSequenceClassification, DistilBertTokenizerFast,
        Trainer, TrainingArguments,
    )

    label_list = sorted(set(labels))
    label_to_id = {label: i for i, label in enumerate(label_list)}

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    encodings = tokenizer(texts, truncation=True, padding=True)
    label_ids = [label_to_id[label] for label in labels]

    class _EncodedDataset(Dataset):
        def __len__(self):
            return len(label_ids)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in encodings.items()}
            item['labels'] = torch.tensor(label_ids[idx])
            return item

    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_list))

    training_args = TrainingArguments(
        output_dir=output_dir or os.path.join(paths.REPORTS_DIR, 'distilbert_checkpoints'),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        seed=seed,
        logging_strategy='no',
        save_strategy='no',
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=_EncodedDataset())
    trainer.train()

    return {'model': model, 'tokenizer': tokenizer, 'label_list': label_list}


def predict(fitted: dict, texts: list[str]) -> list[str]:
    import torch

    model = fitted['model']
    tokenizer = fitted['tokenizer']
    label_list = fitted['label_list']

    model.eval()
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        logits = model(**encodings).logits
    predicted_ids = logits.argmax(dim=-1).tolist()
    return [label_list[i] for i in predicted_ids]
