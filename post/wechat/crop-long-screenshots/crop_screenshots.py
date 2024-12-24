import os
import cv2

def crop_screenshots(input_folder, output_folder, crop_ratio=(9, 12), margin=50, start_pixel=0):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all images in the input folder
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        
        # Skip non-image files
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Load the image
        img = cv2.imread(input_path)
        if img is None:
            print(f"Unable to read image: {file_name}")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Calculate crop height based on the ratio
        crop_height = int((crop_ratio[1] / crop_ratio[0]) * img_width)
        
        output_subfolder = os.path.join(output_folder, os.path.splitext(file_name)[0])
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Start cropping after the specified start_pixel
        start_y = start_pixel
        counter = 1
        
        while start_y < img_height:
            # Define crop area
            end_y = min(start_y + crop_height, img_height)  # Ensure end_y doesn't exceed img height
            
            # Crop the image
            cropped_img = img[start_y:end_y, :]
            
            # Save the cropped image
            output_file = os.path.join(output_subfolder, f"crop_{counter:03d}.png")
            cv2.imwrite(output_file, cropped_img)
            
            # If we are at the bottom of the image, break the loop
            if end_y == img_height:
                break
            
            # Update starting point with margin
            start_y = end_y - margin
            counter += 1
        
        print(f"Processed: {file_name}, {counter-1} sub-images saved.")
    print("Processing complete.")

# Usage example
if __name__ == "__main__":
    input_folder = "//home/m/Documents/github/blog-resources/content/post/wechat/crop-long-screenshots/taxreport"    # Input folder path
    output_folder = input_folder + "/3:4" # Output folder path
    crop_ratio = (3, 4)                 # Desired crop ratio (width:height)
    margin = 50                         # Margin in pixels
    start_pixel = 200                   # Starting pixel from the top

    crop_screenshots(input_folder, output_folder, crop_ratio, margin, start_pixel)
