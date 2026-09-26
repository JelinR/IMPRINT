import os
import time
import hashlib
from tqdm import tqdm
import cv2
import shutil

import logging
logging.getLogger().setLevel(logging.CRITICAL)

try:
    from icrawler.builtin import GoogleImageCrawler
except:
    print(f"Could not Load icrawler library. This is fine if you are not scraping the net.")

SCRAPED_SAVE_DIR = r"data/scraped_imgs/junk"


def run_crawler(query, num_images, save_folder, file_idx_offset):
    crawler = GoogleImageCrawler(storage={'root_dir': save_folder})
    crawler.crawl(keyword=query, max_num=num_images, file_idx_offset=file_idx_offset)

def get_unique_count_and_remove_duplicates(save_folder):
    unique_hashes = {}
    files = [f for f in os.listdir(save_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for f in files:
        path = os.path.join(save_folder, f)
        try:
            with open(path, 'rb') as file:
                file_hash = hashlib.md5(file.read()).hexdigest()
            if file_hash in unique_hashes:
                os.remove(path)
            else:
                unique_hashes[file_hash] = path
        except Exception as e:
            print(f"Error processing file {f}: {e}")
    return len(unique_hashes)

def download_google_images(query, num_images=5, save_dir = SCRAPED_SAVE_DIR, extra_tries = 1):
    current_path = os.getcwd()
    query_dir = os.path.join(current_path, save_dir, query)
    os.makedirs(query_dir, exist_ok=True)

    unique_count = get_unique_count_and_remove_duplicates(query_dir)
    file_idx_offset = unique_count
    pbar = tqdm(total=num_images, desc=f"Downloading {query}", unit="img")
    pbar.update(unique_count)

    n_tries = 0
    while unique_count < num_images:
        remaining = num_images - unique_count
        run_crawler(query, remaining, query_dir, file_idx_offset)
        time.sleep(2)
        new_unique_count = get_unique_count_and_remove_duplicates(query_dir)
        pbar.update(new_unique_count - unique_count)

        if new_unique_count == unique_count:
            n_tries += 1
            if n_tries >= extra_tries:
                print(f"No image could be found! Aborting...")
                return

        unique_count = new_unique_count
        file_idx_offset = unique_count

    pbar.close()
    print(f"Download complete for query: {query}! Saved at: {query_dir}")

def load_images(query: str, num_images: int, save_dir: str = SCRAPED_SAVE_DIR):
    
    save_dir = os.path.join( os.getcwd(), save_dir )
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n Scrape Query: {repr(query)}")
    
    is_file_present = (query in os.listdir(save_dir))
    if not is_file_present:
        # print(f"{query} files not found! Downloading {num_images} images...")
        # download_google_images(query, num_images, save_dir)
        print(f"{query} files not found in {save_dir}! Skipping...")
        return []
    
    query_dir = os.path.join(save_dir, query)
    imgs = [cv2.imread( os.path.join(query_dir, img_path) )[:, :, ::-1] for img_path in os.listdir(query_dir)]

    # if len(imgs) < num_images:
    #     n_imgs_left = num_images - len(imgs)
    #     print(f"{num_images} images not found for {query}, requires {n_imgs_left} more images! Removing directory and Downloading {num_images} images...")
        
    #     shutil.rmtree(query_dir)
    #     download_google_images(query, num_images, save_dir)

    #     imgs = [cv2.imread( os.path.join(query_dir, img_path) )[:, :, ::-1] for img_path in os.listdir(query_dir)]
    
    imgs = imgs[:num_images]    
    if len(imgs) < num_images:
        imgs = []
    
    print(f" Loaded {len(imgs)}/{num_images} images for {query}")

    return imgs



def save_downloaded_urls(url, url_file):
    with open(url_file, "a") as f:
        f.write(url + "\n")

def load_downloaded_urls(url_file):

    if os.path.exists(url_file):
        with open(url_file, 'r') as f:
            return set(f.read().splitlines())
    return set()