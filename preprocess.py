import pandas as pd
import re

def clean_html(raw_html):
    if pd.isna(raw_html): return ""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, ' ', str(raw_html)).strip()

def normalize_size(size_str):
    if not isinstance(size_str, str): return None
    size_str = size_str.lower()
    size_str = size_str.replace("â€²", "'").replace("â€³", '"')
    
    match = re.search(r"(\d+)\s*x\s*(\d+)", size_str)
    if match:
        return f"{match.group(1)}x{match.group(2)}"
    return None

def extract_shape(size_str):
    """Extracts purely geometric shapes. Runner is an aspect ratio, not a shape."""
    if not isinstance(size_str, str): return "rectangle"
    size_str = size_str.lower()
    if "round" in size_str: return "round"
    if "oval" in size_str: return "oval"
    if "square" in size_str: return "square"
    return "rectangle"

def size_bucket(size):
    if not size: return None
    try:
        w, h = map(int, size.split("x"))
        area = w * h
        if area <= 20:
            return "small"
        elif area <= 60:
            return "medium"
        else:
            return "large"
    except Exception:
        return None

def process_catalog(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    cleaned_data = []

    for handle, group in df.groupby('Handle'):
        main_image_rows = group[group['Image Position'] == 1.0]
        primary_row = main_image_rows.iloc[0] if not main_image_rows.empty else group.iloc[0]
            
        title = str(primary_row.get('Title', ''))
        tags = str(primary_row.get('Tags', ''))
        if tags == 'nan': tags = ''
            
        prices = pd.to_numeric(group['Variant Price'], errors='coerce').dropna()
        min_price = prices.min() if not prices.empty else None
        max_price = prices.max() if not prices.empty else None
        
        opt_names = {}
        for i in [1, 2, 3]:
            col = f'Option{i} Name'
            if col in group.columns:
                valid_names = group[col].dropna().astype(str)
                valid_names = valid_names[valid_names != 'nan']
                if not valid_names.empty:
                    opt_names[i] = valid_names.iloc[0].lower()
                else:
                    opt_names[i] = ''
        
        raw_sizes = []
        colors = []
        
        for _, row in group.iterrows():
            for i in [1, 2, 3]:
                name = opt_names.get(i, '')
                val_col = f'Option{i} Value'
                
                if val_col in row:
                    val = str(row[val_col])
                    if val and val.lower() != 'nan':
                        if 'size' in name and val not in raw_sizes:
                            raw_sizes.append(val)
                        elif 'color' in name and val not in colors:
                            colors.append(val)
                            
        if not raw_sizes and 'Option1 Value' in group.columns:
            possible_sizes = group['Option1 Value'].dropna().astype(str)
            raw_sizes = [v for v in possible_sizes if re.search(r"\d+\s*x\s*\d+", v.lower())]
            
        if not colors and 'Option2 Value' in group.columns:
            colors = group['Option2 Value'].dropna().astype(str).tolist()
            
        colors = list(set([c.lower().strip() for c in colors if c.lower().strip() != 'nan']))
        
        normalized_sizes = []
        shapes = set()
        
        for raw_size in raw_sizes:
            shapes.add(extract_shape(raw_size))
            clean_size = normalize_size(raw_size)
            if clean_size and clean_size not in normalized_sizes:
                normalized_sizes.append(clean_size)
        
        # Deduplicate Shapes cleanly without runner
        non_rect_shapes = {"round", "oval", "square"}
        # If the rug has round/oval/square variants, we want to capture that, but 
        # let's just keep the unique set of shapes. It might genuinely be a rectangle main rug with a round variant.
        shapes = list(shapes)

        size_categories = list({size_bucket(s) for s in normalized_sizes if size_bucket(s)})
                
        is_rug_pad = "rug pad" in tags.lower() or "rug pad" in title.lower()
        
        # SEPARATED USAGE AND STYLE EXTRACTORS
        style_types = []
        usage_types = []
        tags_lower = tags.lower()
        title_lower = title.lower()
        
        # 1. Extract Usage
        if "outdoor" in tags_lower or "outdoor" in title_lower:
            usage_types.append("outdoor")
            
        # 2. Extract Style purely
        if "vintage" in tags_lower or "traditional" in tags_lower: style_types.append("traditional")
        if "modern" in tags_lower or "geometric" in tags_lower: style_types.append("modern")
        if "bohemian" in tags_lower or "boho" in tags_lower: style_types.append("bohemian")
        if "persian" in tags_lower: style_types.append("persian")
        
        if not style_types:
            if "vintage" in title_lower or "traditional" in title_lower: style_types.append("traditional")
            if "modern" in title_lower or "geometric" in title_lower: style_types.append("modern")
            if "bohemian" in title_lower or "boho" in title_lower: style_types.append("bohemian")
            if "persian" in title_lower: style_types.append("persian")
            
        styles = list(set(style_types))
        usages = list(set(usage_types))
            
        # Add both to semantic embedding
        embedding_text = f"{title} {' '.join(colors)} {' '.join(styles)} {' '.join(usages)} {tags}"
        embedding_text = re.sub(r" +", " ", embedding_text).strip()
        
        product = {
            "handle": handle,
            "title": title,
            "image_url": str(primary_row.get('Image Src', '') or ''),
            "min_price": min_price,
            "max_price": max_price,
            "sizes": normalized_sizes,
            "size_categories": size_categories,
            "shapes": shapes,
            "colors": colors,
            "styles": styles,
            "usages": usages, # newly added!
            "tags": tags,
            "is_rug_pad": is_rug_pad,
            "embedding_text": embedding_text
        }
        cleaned_data.append(product)

    cleaned_df = pd.DataFrame(cleaned_data)
    print(f"Processed {len(cleaned_df)} unique products.")
    return cleaned_df

if __name__ == "__main__":
    df = process_catalog('intern_assignment.csv')
    df.to_csv('cleaned_catalog.csv', index=False)
    print("Saved to cleaned_catalog.csv")