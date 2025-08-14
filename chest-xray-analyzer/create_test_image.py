#!/usr/bin/env python3
"""
Creates a test grayscale image that mimics an X-ray for demo purposes.
This is NOT a real X-ray - just for testing the application flow.
"""

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def create_demo_xray(filename="test_xray.png"):
    # Create a grayscale image
    width, height = 512, 512
    
    # Create base gradient (darker at edges, lighter in center)
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # Draw gradient circles to simulate chest cavity
    center_x, center_y = width // 2, height // 2
    
    for i in range(255, 0, -2):
        radius = int((i / 255) * min(width, height) / 2)
        left = center_x - radius
        top = center_y - radius
        right = center_x + radius
        bottom = center_y + radius
        
        # Vary intensity to create depth
        intensity = int(i * 0.6)
        draw.ellipse([left, top, right, bottom], fill=intensity)
    
    # Add some "lung" shapes
    lung_left = Image.new('L', (width, height), 0)
    draw_left = ImageDraw.Draw(lung_left)
    draw_left.ellipse([width//4-80, height//3-50, width//4+80, height//3+150], fill=180)
    
    lung_right = Image.new('L', (width, height), 0)
    draw_right = ImageDraw.Draw(lung_right)
    draw_right.ellipse([3*width//4-80, height//3-50, 3*width//4+80, height//3+150], fill=180)
    
    # Composite the images
    img = Image.blend(img, lung_left, 0.3)
    img = Image.blend(img, lung_right, 0.3)
    
    # Add some noise for realism
    img_array = np.array(img)
    noise = np.random.normal(0, 10, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Apply slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Add "spine" shadow
    draw = ImageDraw.Draw(img)
    draw.rectangle([center_x-15, 50, center_x+15, height-50], fill=100)
    
    # Save the image
    img.save(filename)
    print(f"✅ Created demo X-ray image: {filename}")
    print("📌 Note: This is a synthetic image for testing only, not a real medical X-ray")
    return filename

if __name__ == "__main__":
    create_demo_xray()
    print("\n🚀 You can now:")
    print("1. Start the backend: cd backend && uvicorn main:app --reload")
    print("2. Open frontend/index.html in your browser")
    print("3. Upload the test_xray.png file to see the analysis!")