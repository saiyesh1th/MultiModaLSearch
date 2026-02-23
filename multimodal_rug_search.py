import os
import re
import ast
import time
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel


# ==========================================================
# CONFIG
# ==========================================================

CATALOG_FILE = "cleaned_catalog.csv"
TEXT_EMBED_FILE = "product_text_embeddings.npy"
IMAGE_EMBED_FILE = "product_image_embeddings.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_WEIGHT = 0.7
TEXT_WEIGHT = 0.3


# ==========================================================
# LOAD MODELS (Load Once)
# ==========================================================

TEXT_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

CLIP_MODEL = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)

CLIP_PROCESSOR = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)


# ==========================================================
# LOAD CATALOG
# ==========================================================

def load_catalog(path):
    df = pd.read_csv(path)

    list_cols = [
        "sizes", "size_categories", "shapes",
        "colors", "styles", "usages"
    ]

    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )

    return df


# ==========================================================
# QUERY PARSING
# ==========================================================

COMMON_COLORS = [
    "ivory", "beige", "blue", "navy", "gray", "grey",
    "black", "white", "red", "green", "brown",
    "cream", "tan", "gold", "pink", "teal", "charcoal"
]

COMMON_STYLES = [
    "traditional", "modern", "bohemian",
    "boho", "persian", "abstract",
    "contemporary", "casual"
]


def parse_query(query):
    query = query.lower()

    size_match = re.search(r"(\d+)\s*x\s*(\d+)", query)
    size = f"{size_match.group(1)}x{size_match.group(2)}" if size_match else None

    shape = next((s for s in ["round", "runner", "oval", "square"] if s in query), None)
    color = next((c for c in COMMON_COLORS if c in query), None)

    style = None
    for s in COMMON_STYLES:
        if s in query:
            style = "bohemian" if s == "boho" else s
            break

    size_category = next((sc for sc in ["small", "medium", "large"] if sc in query), None)

    return {
        "size": size,
        "shape": shape,
        "color": color,
        "style": style,
        "size_category": size_category
    }


# ==========================================================
# PRECOMPUTE EMBEDDINGS
# ==========================================================

def compute_text_embeddings(df):
    embeddings = TEXT_MODEL.encode(
        df["embedding_text"].tolist(),
        normalize_embeddings=True
    )
    np.save(TEXT_EMBED_FILE, embeddings)


def compute_image_embeddings(df):
    image_embeddings = []

    for idx, row in df.iterrows():
        try:
            image = None
            for attempt in range(3):
                try:
                    response = requests.get(row["image_url"], timeout=15)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    break
                except Exception:
                    time.sleep(2)

            if image is None:
                raise ValueError("Image load failed")

            inputs = CLIP_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                features = CLIP_MODEL.get_image_features(**inputs)

            if not hasattr(features, "detach"):
                features = features.pooler_output

            features = features.detach().cpu().numpy()[0]
            embedding = features / np.linalg.norm(features)

            image_embeddings.append(embedding)

        except Exception:
            image_embeddings.append(np.random.normal(size=512))

    np.save(IMAGE_EMBED_FILE, np.array(image_embeddings))


# ==========================================================
# EXPLANATION GENERATION
# ==========================================================

def generate_explanation(row, parsed_query):
    reasons = []

    if parsed_query["color"] and parsed_query["color"] in row["colors"]:
        reasons.append(f"matches requested color ({parsed_query['color']})")

    if parsed_query["style"] and parsed_query["style"] in row["styles"]:
        reasons.append(f"aligns with {parsed_query['style']} style")

    if parsed_query["size"] and parsed_query["size"] in row["sizes"]:
        reasons.append(f"available in {parsed_query['size']} size")

    if not reasons:
        reasons.append("is visually similar to the provided room")

    return "Recommended because it " + ", ".join(reasons) + "."


# ==========================================================
# STRUCTURED SEARCH
# ==========================================================

def structured_search(query, df, text_embeddings,
                      min_price=None, max_price=None,
                      top_k=5):

    parsed = parse_query(query)

    query_embedding = TEXT_MODEL.encode(
        [query],
        normalize_embeddings=True
    )

    metadata_scores = []

    for _, row in df.iterrows():

        if row["is_rug_pad"] and "pad" not in query.lower():
            metadata_scores.append(0)
            continue

        # Price filter
        if min_price and row["max_price"] < min_price:
            metadata_scores.append(0)
            continue

        if max_price and row["min_price"] > max_price:
            metadata_scores.append(0)
            continue

        score = 0

        if parsed["size"] and parsed["size"] in row["sizes"]:
            score += 3

        if parsed["size_category"] and parsed["size_category"] in row["size_categories"]:
            score += 2

        if parsed["shape"] and parsed["shape"] in row["shapes"]:
            score += 2

        if parsed["color"] and parsed["color"] in row["colors"]:
            score += 2

        if parsed["style"] and parsed["style"] in row["styles"]:
            score += 2

        metadata_scores.append(score)

    metadata_scores = np.array(metadata_scores)

    if metadata_scores.max() > 0:
        metadata_scores = metadata_scores / metadata_scores.max()

    semantic_scores = cosine_similarity(query_embedding, text_embeddings)[0]

    final_scores = 0.6 * semantic_scores + 0.4 * metadata_scores

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["final_score"] = final_scores[top_indices]
    results["explanation"] = results.apply(
        lambda row: generate_explanation(row, parsed), axis=1
    )

    return results[["handle", "title", "image_url", "final_score", "explanation"]]


# ==========================================================
# MULTIMODAL SEARCH
# ==========================================================

def multimodal_search(room_image_path,
                      df,
                      image_embeddings,
                      optional_text=None,
                      text_embeddings=None,
                      top_k=5):

    df_filtered = df[df["is_rug_pad"] == False].reset_index(drop=True)
    image_embeddings_filtered = image_embeddings[df["is_rug_pad"] == False]

    image = Image.open(room_image_path).convert("RGB")

    inputs = CLIP_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        features = CLIP_MODEL.get_image_features(**inputs)

    if not hasattr(features, "detach"):
        features = features.pooler_output

    features = features.detach().cpu().numpy()[0]
    room_embedding = features / np.linalg.norm(features)

    image_sim = cosine_similarity([room_embedding], image_embeddings_filtered)[0]

    if optional_text and text_embeddings is not None:
        text_emb = TEXT_MODEL.encode(
            [optional_text],
            normalize_embeddings=True
        )
        text_sim = cosine_similarity(text_emb, text_embeddings[df["is_rug_pad"] == False])[0]
        final_scores = IMAGE_WEIGHT * image_sim + TEXT_WEIGHT * text_sim
    else:
        final_scores = image_sim

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = df_filtered.iloc[top_indices].copy()
    results["final_score"] = final_scores[top_indices]
    results["explanation"] = "Recommended due to visual similarity to the room."

    return results[["handle", "title", "image_url", "final_score", "explanation"]]