# convert_kaggle_styles_to_labels.py

import pandas as pd
import os

# ---------- CONFIGURATION ----------
CSV_INPUT = "styles.csv"
IMAGES_DIR = "images"  # folder containing the images
OUTPUT_CSV = "labels.csv"

# ---- Define your target attributes ----
colors_list = ["black", "white", "red", "blue", "green", "yellow", "purple", "brown"]
article_list = ["tshirt", "shirt", "jeans", "shorts", "dress", "jacket", "sweatshirt", "kurta", "top"]

attribute_names = colors_list + article_list

def main():
    df = pd.read_csv(CSV_INPUT, engine="python", on_bad_lines='skip')

    # Build image filename column 
    df["image_name"] = df["id"].astype(str) + ".jpg"
    df = df[df["image_name"].apply(lambda x: os.path.exists(os.path.join(IMAGES_DIR, x)))]

    # Initialize zeros for attribute columns
    for attr in attribute_names:
        df[attr] = 0

    # Populate values
    for idx, row in df.iterrows():
        color = str(row["baseColour"]).lower() if not pd.isna(row["baseColour"]) else ""
        article = str(row["articleType"]).lower() if not pd.isna(row["articleType"]) else ""
        prod_name = str(row["productDisplayName"]).lower() if not pd.isna(row["productDisplayName"]) else ""

        # Colors
        for c in colors_list:
            if c in color or c in prod_name:
                df.at[idx, c] = 1

        # Article Types
        for a in article_list:
            if a in article or a in prod_name:
                df.at[idx, a] = 1

    # Keep only image + attributes
    final_cols = ["image_name"] + attribute_names
    df_final = df[final_cols]
    df_final.to_csv(OUTPUT_CSV, index=False)
    print("Saved labels to", OUTPUT_CSV)
    print("Number of rows:", len(df_final))

if __name__ == "__main__":
    main()