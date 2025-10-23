from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from urllib.request import urlretrieve
from pathlib import Path
import os
import pandas as pd

def download_image(image_link, savefolder, filename):
    """Download a single image and save it with a custom filename."""
    if isinstance(image_link, str):
        filename = str(filename)
        # Ensure the filename has .jpg extension
        if not filename.endswith(".jpg"):
            filename = f"{filename}.jpg"
        path = os.path.join(savefolder, filename)
        try:
            urlretrieve(image_link, path)
        except Exception as e:
            print(f"Warning: {image_link} -> {e}")

def download_images(df, download_folder, max_workers=50):
    """Download images using sample_id as filenames."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Prepare list of tuples (url, filename)
    tasks = list(zip(df["image_link"], df["sample_id"]))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use lambda to unpack each tuple
        list(tqdm(executor.map(lambda args: download_image(args[0], download_folder, args[1]), tasks),
                  total=len(tasks)))

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example DataFrame
    df = pd.read_csv("train.csv")
    download_folder = "AMLC/train"
    download_images(df, download_folder, max_workers=40)
    df = pd.read_csv("test.csv")
    download_folder = "AMLC/test"
    download_images(df, download_folder, max_workers=40)
