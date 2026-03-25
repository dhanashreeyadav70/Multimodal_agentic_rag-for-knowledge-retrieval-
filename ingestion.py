import json
from langchain_core.documents import Document


def flatten_json(data, parent_key=""):

    items = []

    for k, v in data.items():

        new_key = f"{parent_key}.{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key).items())

        else:
            items.append((new_key, v))

    return dict(items)


def convert_record_to_text(record):

    flattened = flatten_json(record)

    parts = []

    for key, value in flattened.items():

        if value is None:
            continue

        val = str(value).strip()

        if val == "":
            continue

        parts.append(f"{key}: {val}")

    return "\n".join(parts)


import json
from langchain_core.documents import Document


def load_json(file_path):

    with open(file_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    documents = []

    for record in data:

        text = "\n".join([f"{k}: {v}" for k, v in record.items() if v])

        doc = Document(
            page_content=text,
            metadata={"record_id": record.get("zlid", None)}
        )

        documents.append(doc)

    return documents