from pathlib import Path
from duckduckgo_search import DDGS
from random import shuffle
from PIL import Image, UnidentifiedImageError
import requests
import hashlib

def download_class_images(query, max_results=5, output_dir='images', split=0.8) -> None:
    """Download images for a specific class or extend and already downloaded dataset.

    Args:
        query (str): The search query for the image class.
        max_results (int, optional): The maximum number of images to download. Defaults to 5.
        output_dir (str, optional): The directory to save the images. Defaults to 'images'.
        split (float, optional): The train/validation split ratio. Defaults to 0.8.
        
    This function uses the DuckDuckGo search API to find images related to the query,
    downloads them, and organizes them into 'train' and 'val' subdirectories within the
    specified output directory. The images are named using the query and an index.
    The function ensures that the output directory exists and creates the necessary subdirectories.
    The images are downloaded with a timeout to handle network issues, and any errors during download
    are printed to the console. The function also shuffles the results before splitting them into training
    and validation sets.
    The images are saved in JPEG format with a naming convention that includes the query and an index with offset
    to ensure that images from different queries do not overwrite each other. This is particularly useful when we want to expand the already downloaded dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = DDGS().images(query, max_results=max_results)
    
    train_dir = output_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir = output_dir / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)
    
    split_index = int(len(results) * split)
    shuffle(results)
    class_name = query.lower()
    
    train_dir = train_dir / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir = val_dir / class_name
    val_dir.mkdir(parents=True, exist_ok=True)
    
    train_offset = len(list(train_dir.glob('*.jpg')))
    cnt = 0
    for i, item in enumerate(results[:split_index], start=train_offset):
        image_url = item['image']
        image_path = train_dir / f"{query}_{i+1}.jpg"
        
        try:
            response = requests.get(image_url, timeout=5)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Skipped {image_url} due to error: {e}")
            cnt += 1
                
    val_offset = len(list(val_dir.glob('*.jpg')))
    for i, item in enumerate(results[split_index:], start=val_offset):
        image_url = item['image']
        image_path = val_dir / f"{query}_{i+1}.jpg"
        
        try:
            response = requests.get(image_url, timeout=5)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Skipped {image_url} due to error: {e}")
            cnt += 1
                
    print(f"Downloaded {len(results) - cnt} images for class '{class_name}'")

def remove_corrupted_images(directory) -> None:
    """Remove corrupted images (.jpg) from a directory.

    Args:
        directory (str): The directory to scan for corrupted images.
        
    This function attempts to open each image file and verifies if it is a valid image.
    If an image cannot be opened or is corrupted, it is removed from the directory.
    It prints the number of corrupted images removed.
    """
    directory = Path(directory)
    
    cnt = 0
    for image_path in directory.rglob('*.jpg'):
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify that it is an image
        except (UnidentifiedImageError, OSError):
            print(f"Removing corrupted image: {image_path}")
            image_path.unlink()
            cnt += 1
            
    print(f"Removed {cnt} corrupted images from {directory}")
 
def remove_duplicate_images(directory) -> None:
    """Remove duplicate images (.jpg) from a directory based on their MD5 hash.

    Args:
        directory (str): the target directory to scan for duplicate images.
    This function computes the MD5 hash of each image file and removes duplicates based on the hash
    It removes Corrupted images first to ensure that only valid images are processed.
    It prints the number of duplicates removed.
    """
    remove_corrupted_images(directory)  # Clean up corrupted images first
    directory = Path(directory)
    hashes = set()
    duplicates = 0
    
    for image_path in directory.rglob('*.jpg'):
        dup = False
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in hashes:
                duplicates += 1
                dup = True
            else:
                hashes.add(file_hash)
                
        if dup:    
            image_path.unlink()
            print(f"Duplicate found and removed: {image_path}")
    
    print(f"Removed {duplicates} duplicate images from {directory}") 