from PIL import Image
import os

def combine_webp_images():
    # Get all webp files in current directory
    webp_files = [f for f in os.listdir('.') if f.endswith('.webp')]
    
    if not webp_files:
        print("No WEBP files found in current directory")
        return
    
    # Open all images
    images = [Image.open(f) for f in webp_files]
    
    # Calculate total height and max width
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    
    # Create new image with combined dimensions
    combined = Image.new('RGBA', (max_width, total_height))
    
    # Paste images one below another
    y_offset = 0
    for img in images:
        combined.paste(img, (0, y_offset))
        y_offset += img.height
    
    # Save combined image
    combined.save('combined_output.webp', 'WEBP')
    print(f"Combined {len(webp_files)} images into combined_output.webp")

if __name__ == "__main__":
    combine_webp_images()
