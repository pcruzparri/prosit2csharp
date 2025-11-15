import requests
import os
import zipfile
from pathlib import Path
import shutil

def download_prosit_model():
    """
    Download and decompress the Prosit model from Figshare.
    """
    # URL for the Prosit model
    model_url = "https://figshare.com/ndownloader/files/24635243"
    
    # Create models directory if it doesn't exist
    models_dir = Path("./models").absolute()
    models_dir.mkdir(exist_ok=True)
    
    # Define file paths
    zip_file_path = models_dir / "prosit_model.zip"
    
    print("Downloading Prosit model...")
    print(f"URL: {model_url}")
    print(f"Destination: {zip_file_path}")
    
    try:
        # Download the file
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        file_size = int(response.headers.get('content-length', 0))
        if file_size > 0:
            print(f"File size: {file_size / (1024*1024):.1f} MB")
        
        # Download with progress indication
        downloaded_size = 0
        with open(zip_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if file_size > 0:
                        progress = (downloaded_size / file_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)
        
        print("\nDownload completed!")
        
        # Decompress the zip file
        print("Decompressing...")
        extract_dir = models_dir / "prosit"
        
        # Remove existing directory if it exists
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"Model extracted to: {extract_dir}")
        
        # List extracted files
        print("\nExtracted files:")
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(extract_dir)
                file_size_mb = file_path.stat().st_size / (1024*1024)
                print(f"  {relative_path} ({file_size_mb:.1f} MB)")
        
        # Optionally remove the zip file to save space
        print(f"\nRemoving zip file: {zip_file_path}")
        zip_file_path.unlink()
        
        print("Prosit model download and extraction completed successfully!")
        return extract_dir
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None
    except zipfile.BadZipFile as e:
        print(f"Error extracting zip file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    download_prosit_model()
