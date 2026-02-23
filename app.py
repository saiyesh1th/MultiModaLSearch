import os
import streamlit as st
import numpy as np

from multimodal_rug_search import (
    load_catalog,
    structured_search,
    multimodal_search,
    compute_text_embeddings,
    compute_image_embeddings,
    parse_query,
    CATALOG_FILE,
    TEXT_EMBED_FILE,
    IMAGE_EMBED_FILE
)

st.set_page_config(page_title="Multimodal Rug Search", layout="wide")

st.title("🧩 Multimodal Rug Search & Recommendation System")
st.caption("Fusion Formula: 0.7 × Image Similarity + 0.3 × Text Similarity")

# ==========================================================
# LOAD SYSTEM (CACHED)
# ==========================================================

@st.cache_resource
def load_system():
    catalog = load_catalog(CATALOG_FILE)

    if not os.path.exists(TEXT_EMBED_FILE):
        compute_text_embeddings(catalog)

    if not os.path.exists(IMAGE_EMBED_FILE):
        compute_image_embeddings(catalog)

    text_embeddings = np.load(TEXT_EMBED_FILE)
    image_embeddings = np.load(IMAGE_EMBED_FILE)

    return catalog, text_embeddings, image_embeddings

catalog, text_embeddings, image_embeddings = load_system()

# ==========================================================
# MODE SELECTION
# ==========================================================

mode = st.radio(
    "Choose Mode:",
    ["Structured Text Search", "Image Search", "Image + Text Search"]
)

st.markdown("---")

# ==========================================================
# STRUCTURED SEARCH
# ==========================================================

if mode == "Structured Text Search":

    query = st.text_input("Enter rug query")

    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("Min Price", min_value=0, value=0)
    with col2:
        max_price = st.number_input("Max Price", min_value=0, value=1000)

    if st.button("Search Structured"):
        if query.strip() == "":
            st.warning("Please enter a query.")
        else:
            parsed = parse_query(query)
            st.write("Parsed Query:", parsed)

            results = structured_search(
                query,
                catalog,
                text_embeddings,
                min_price=min_price,
                max_price=max_price
            )

            st.subheader("Top Results")
            for _, row in results.iterrows():
                col_img, col_text = st.columns([1,2])
                with col_img:
                    st.image(row["image_url"])
                with col_text:
                    st.write(f"### {row['title']}")
                    st.write(f"Score: {round(row['final_score'],3)}")
                    st.write(row["explanation"])
                st.markdown("---")

# ==========================================================
# IMAGE SEARCH
# ==========================================================

if mode == "Image Search":

    st.subheader("Upload Room Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        key="image_only_uploader"
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Room", use_column_width=True)

        with open("temp_room.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Run Image Search"):
            results = multimodal_search(
                "temp_room.jpg",
                catalog,
                image_embeddings
            )

            st.subheader("Top Recommendations")

            for _, row in results.iterrows():
                col_img, col_text = st.columns([1,2])
                with col_img:
                    st.image(row["image_url"])
                with col_text:
                    st.write(f"### {row['title']}")
                    st.write(f"Score: {round(row['final_score'],3)}")
                    st.write(row["explanation"])
                st.markdown("---")

# ==========================================================
# IMAGE + TEXT SEARCH
# ==========================================================

if mode == "Image + Text Search":

    st.subheader("Upload Room Image + Optional Text")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        key="image_text_uploader"
    )

    optional_text = st.text_input("Optional text (e.g., modern neutral)")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Room", use_column_width=True)

        with open("temp_room.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Run Multimodal Search"):
            results = multimodal_search(
                "temp_room.jpg",
                catalog,
                image_embeddings,
                optional_text=optional_text,
                text_embeddings=text_embeddings
            )

            st.subheader("Top Recommendations")

            for _, row in results.iterrows():
                col_img, col_text = st.columns([1,2])
                with col_img:
                    st.image(row["image_url"])
                with col_text:
                    st.write(f"### {row['title']}")
                    st.write(f"Score: {round(row['final_score'],3)}")
                    st.write(row["explanation"])
                st.markdown("---")