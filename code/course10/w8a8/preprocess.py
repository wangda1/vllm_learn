from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Pick any chat model tokenizer that provides a chat template.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PREVIEW_CHARS = 300
PREVIEW_TOKEN_IDS = 20


def preprocess(example, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


def tokenize(sample, tokenizer):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


def shorten(text, max_chars=PREVIEW_CHARS):
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load the full split, shuffle it, then select 512 samples.
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    print("=== Step 1: Raw sample ===")
    print("columns:", ds.column_names)
    print("messages count:", len(ds[0]["messages"]))
    first_message = ds[0]["messages"][0]
    print("first message role:", first_message["role"])
    print("first message preview:", shorten(first_message["content"]))

    ds = ds.map(lambda example: preprocess(example, tokenizer))

    print("\n=== Step 2: After chat template ===")
    print("columns:", ds.column_names)
    print("text preview:", shorten(ds[0]["text"]))

    ds = ds.map(
        lambda sample: tokenize(sample, tokenizer),
        remove_columns=ds.column_names,
    )

    print("\n=== Step 3: After tokenization ===")
    print("columns:", ds.column_names)
    print("input_ids length:", len(ds[0]["input_ids"]))
    print("input_ids preview:", ds[0]["input_ids"][:PREVIEW_TOKEN_IDS])
    if "attention_mask" in ds[0]:
        print("attention_mask preview:", ds[0]["attention_mask"][:PREVIEW_TOKEN_IDS])


if __name__ == "__main__":
    main()
