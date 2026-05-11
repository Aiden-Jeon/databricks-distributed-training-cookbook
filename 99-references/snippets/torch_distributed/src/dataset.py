import uuid
from datasets import load_dataset as load_huggingface_dataset
from transformers import AutoTokenizer
from streaming import StreamingDataset, StreamingDataLoader


def load_torch_dataset(dataset_name: str, tokenizer_name: str):
    tokenized_datasets = load_tokenized_dataset(dataset_name, tokenizer_name)
    tokenized_datasets.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"]
    )
    return tokenized_datasets


def load_tokenized_dataset(dataset_name, tokenizer_name):
    dataset = load_huggingface_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, return_tensors="pt"
        )

    tokenized_datasets = dataset.map(_tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns("text")
    return tokenized_datasets


def load_mds_dataset(path, batch_size, label, use_local=False):
    random_uuid = uuid.uuid4()
    local_path = f"/local_disk0/{random_uuid}"
    print(f"Getting {label} data from UC Volumes at {path} and saving to {local_path}")
    if use_local:
        dataset = StreamingDataset(
            remote=path,
            local=local_path,
            shuffle=False,
            batch_size=batch_size,
        )
    else:
        dataset = StreamingDataset(
            local=path,
            shuffle=False,
            batch_size=batch_size,
        )
    return dataset


def get_mds_dataloader(path, batch_size, label, use_local=False):
    dataset = load_mds_dataset(path, batch_size, label, use_local)
    dataloader = StreamingDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )
    return dataloader
