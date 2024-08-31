# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


def predict(text):

    inputs = tokenizer(text, return_tensors="pt", max_length=16384, truncation=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    rates = logits.softmax(dim=1).tolist()

    df = pd.DataFrame(rates, columns=model.config.id2label.values())
    return df


data = pd.read_csv("data/MSFT/news/MSFT.csv").iloc[:5]


sentiment_df = pd.DataFrame(columns=model.config.id2label.values())

for i in tqdm(range(0, len(data))):

    text = data["Title"][i]

    sentiment = predict(text)

    sentiment_df = pd.concat([sentiment_df, sentiment], axis=0, ignore_index=True)


sentiment_df["Date"] = data["Date"]

sentiment_df.to_csv("deneme_analiz.csv", index=False)
"""
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


def predict(text):

    # Split text into chunks of 512 tokens
    inputs = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    rates = logits.softmax(dim=1).tolist()

    df = pd.DataFrame(rates, columns=model.config.id2label.values())
    return df


def split_into_chunks(text, tokenizer, chunk_size=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = torch.cat(
                [chunk, torch.zeros(chunk_size - len(chunk), dtype=torch.long)]
            )
        chunks.append(chunk)
    return chunks


data = pd.read_csv("sentiment_processes/content_extracted.csv").iloc[:10]

sentiment_df = pd.DataFrame(columns=model.config.id2label.values())

for i in tqdm(range(0, len(data))):
    if pd.isnull(data["Content"][i]):
        text = data["Title"][i]
    else:
        text = data["Content"][i]

    # Process text in chunks if it is too long
    chunks = split_into_chunks(text, tokenizer)
    sentiment_chunk_df = pd.DataFrame(columns=model.config.id2label.values())

    for chunk in chunks:
        inputs = {
            "input_ids": chunk.unsqueeze(0),
            "attention_mask": torch.ones_like(chunk).unsqueeze(0),
        }
        with torch.no_grad():
            logits = model(**inputs).logits
        rates = logits.softmax(dim=1).tolist()
        chunk_df = pd.DataFrame(rates, columns=model.config.id2label.values())
        sentiment_chunk_df = pd.concat(
            [sentiment_chunk_df, chunk_df], axis=0, ignore_index=True
        )

    # Average the results from the chunks
    sentiment = sentiment_chunk_df.mean().to_frame().T
    sentiment_df = pd.concat([sentiment_df, sentiment], axis=0, ignore_index=True)

date = data["Date"]
sentiment_df["Date"] = date

sentiment_df.to_csv(
    "sentiment_processes/sentiment_analysed_result_with_content.csv", index=False
)"""


"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


def model_predict(text):
    inputs = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    rates = logits.softmax(dim=1).tolist()
    df = pd.DataFrame(rates, columns=model.config.id2label.values())
    return df


def split_into_chunks(text, tokenizer, chunk_size=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        current_chunk.append(token)
        current_length += 1

        if current_length >= chunk_size - 2:
            chunks.append(
                [tokenizer.cls_token_id] + current_chunk + [tokenizer.sep_token_id]
            )
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(
            [tokenizer.cls_token_id] + current_chunk + [tokenizer.sep_token_id]
        )

    return chunks


def decode_chunk(chunk, tokenizer):
    return tokenizer.decode(chunk, skip_special_tokens=True)


def predict(text):
    chunks = split_into_chunks(text, tokenizer)
    sentiment_chunk_df = pd.DataFrame(columns=model.config.id2label.values())

    for chunk in chunks:
        decoded_chunk = decode_chunk(chunk, tokenizer)
        sentiment = model_predict(decoded_chunk)

        if sentiment_chunk_df.empty:
            sentiment_chunk_df = sentiment
        else:
            sentiment_chunk_df = pd.concat(
                [sentiment_chunk_df, sentiment], axis=0, ignore_index=True
            )

    sentiment = sentiment_chunk_df.mean().to_frame().T
    return sentiment


data = pd.read_csv("sentiment_processes/content_extracted.csv")

sentiment_df = pd.DataFrame(columns=model.config.id2label.values())

for i in tqdm(range(0, len(data))):
    if pd.isnull(data["Content"][i]):
        text = data["Title"][i]
    else:
        text = data["Content"][i]

    sentiment = predict(text)
    if i == 0:
        sentiment_df = sentiment
    else:
        sentiment_df = pd.concat([sentiment_df, sentiment], axis=0, ignore_index=True)

sentiment_df["Date"] = data["Date"]

sentiment_df.to_csv(
    "sentiment_processes/sentiment_analysed_result_with_content.csv", index=False
)
"""
