# 🧩 Multimodal Rug Search & Recommendation System

A multimodal retrieval system that recommends rugs using:

- ✅ Structured text queries  
- ✅ Room image similarity  
- ✅ Image + text fusion  
- ✅ Price filtering  
- ✅ Explainable recommendations  

This project combines rule-based parsing, semantic embeddings, visual similarity (CLIP), and weighted fusion to generate ranked recommendations.

---

## 🚀 Features

### 1️⃣ Structured Text Search
Example:
> 8x10 beige traditional rug

- Size extraction (regex)
- Color detection
- Style matching
- Shape matching
- Size category detection
- Metadata scoring
- Semantic similarity ranking
- Optional price filter

---

### 2️⃣ Image-Based Recommendation

Upload a room image to retrieve visually similar rugs using:

- CLIP image embeddings
- Cosine similarity
- Top-K ranking

---

### 3️⃣ Image + Text Fusion

Example:
> Room Image + "modern neutral"

Weighted fusion:
```
Final Score = 0.7 × Image Similarity + 0.3 × Text Similarity
```
Text refines visual context without overpowering it.

---

### 4️⃣ Price Filtering

Optional minimum and maximum price filtering applied before ranking.

---

### 5️⃣ Explainable Recommendations

Each result includes a short explanation:

- Matched color
- Matched size
- Matched style
- Or visual similarity justification

Example:
> Recommended because it matches the requested beige color and traditional style.

---

# 🏗 System Architecture

Pipeline:
```
Input
↓
Parsing / Embeddings
↓
Similarity Search
↓
Fusion
↓
Ranking (Top-K)
↓
Output
```

### Text Branch
User Query  
→ Query Parser  
→ Metadata Scoring  
→ MiniLM Encoder  
→ Text Similarity  

### Image Branch
Room Image  
→ CLIP Encoder  
→ Image Similarity  

Both similarity scores merge in a weighted fusion layer before ranking.

---

# 🧠 Models Used

### 🔹 Text Embeddings
**SentenceTransformers – all-MiniLM-L6-v2**

- Lightweight
- Fast inference
- Strong semantic similarity performance

### 🔹 Image Embeddings
**CLIP (ViT-B/32)**

- Joint image-text embedding space
- Strong visual representation
- No fine-tuning required

---

# 📊 Ranking Strategy

### Structured Search:
```
Final Score = 0.6 × Semantic Similarity + 0.4 × Metadata Score
```

### Image + Text Search:
```
Final Score = 0.7 × Image Similarity + 0.3 × Text Similarity
```

Top-K rugs are returned after sorting by final score.

---

# ⚠ Failure Cases

### Example:
Room + "pink fluffy rug"

Results remain mostly neutral because:
- Limited pink rugs in catalog
- Image similarity dominates
- Texture descriptors ("fluffy") are not well captured in embeddings

This highlights realistic limitations of embedding-based retrieval systems.

---

# 📁 Project Structure

```
.
├── multimodal_rug_search.py # Core search logic
├── app.py # Streamlit UI
├── cleaned_catalog.csv # Catalog dataset
├── product_text_embeddings.npy # Precomputed text embeddings
├── product_image_embeddings.npy # Precomputed image embeddings
└── README.md
```

---

# ▶️ How to Run

## 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

## 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
If you don't have a requirements file, install manually:
```bash
pip install torch transformers sentence-transformers streamlit pandas numpy pillow scikit-learn requests
```

## 3️⃣ Run Streamlit App
```bash
streamlit run app.py
```
