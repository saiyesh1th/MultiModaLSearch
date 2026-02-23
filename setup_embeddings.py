from multimodal_rug_search import (
    load_catalog,
    compute_text_embeddings,
    compute_image_embeddings
)

CATALOG_FILE = "cleaned_catalog.csv"

catalog = load_catalog(CATALOG_FILE)

print("Generating text embeddings...")
compute_text_embeddings(catalog)

print("Generating image embeddings...")
compute_image_embeddings(catalog)

print("Done. Embeddings saved.")