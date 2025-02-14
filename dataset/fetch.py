import os
import urllib.request
import zipfile

def main():
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    data_dir = "./data"
    
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Tiny ImageNet dataset already downloaded.")

    if not os.path.exists(os.path.join(data_dir, "tiny-imagenet-200")):
        print("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")
    else:
        print("Tiny ImageNet dataset already extracted.")


if __name__ == "__main__":
    main()
