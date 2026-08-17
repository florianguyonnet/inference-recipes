"""Task helpers for the local eval configs (see the yaml files next to this one)."""

import os

LETTERS = "ABCDEFGHIJ"

_MMLU_PRO_INSTRUCTION = (
    'Think step by step, then finish with "the answer is (X)" '
    "where X is the letter of the correct choice."
)


def mmlu_pro_subset(dataset):
    """Every Nth question (MMLU_PRO_STRIDE, default 24 -> 502 of 12032).

    The test split is grouped by category, so `--limit` would only ever hit
    `business`. A fixed stride keeps all 14 categories in proportion and is
    identical from run to run, which is what makes two runs comparable.
    """
    stride = int(os.environ.get("MMLU_PRO_STRIDE", 24))
    return dataset.select(range(0, len(dataset), stride))


def mmlu_pro_doc_to_text(doc) -> str:
    options = "\n".join(
        f"{LETTERS[i]}. {opt.strip()}" for i, opt in enumerate(doc["options"][: len(LETTERS)])
    )
    return f"{doc['question'].strip()}\n{options}\n\n{_MMLU_PRO_INSTRUCTION}"


def aime_doc_to_text(doc) -> str:
    return f"{doc['problem'].strip()}\n\nPut the final answer in \\boxed{{}}."


def gsm8k_doc_to_text(doc) -> str:
    return f"{doc['question'].strip()}\n\nEnd with the final answer as a plain number."


def gsm8k_doc_to_target(doc) -> str:
    return doc["answer"].split("####")[-1].strip()
