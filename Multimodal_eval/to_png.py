import numpy as np
from PIL import Image
import ast
import os

def save_normalized_log_image(log_file_path, output_image_path, line_number=8):
    """
    Reads a specific line with float pixel data from a log file,
    normalizes the data, and saves it as a proper grayscale PNG image.

    Args:
        log_file_path (str): The path to the source log file.
        output_image_path (str): The path to save the output PNG image.
        line_number (int): The line number containing the pixel data (1-based index).
    """
    if not os.path.exists(log_file_path):
        print(f"Error: The file '{log_file_path}' was not found.")
        return

    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < line_number:
            print(f"Error: File has only {len(lines)} lines, cannot read line {line_number}.")
            return

        pixel_line = lines[line_number - 1].strip()
        data_prefix = "pixel_values:"
        if data_prefix not in pixel_line:
            print(f"Error: Could not find '{data_prefix}' on line {line_number}.")
            return

        pixel_data_str = pixel_line.split(data_prefix, 1)[1].strip()
        pixel_list = ast.literal_eval(pixel_data_str)

        # --- NEW: NORMALIZATION STEP ---
        # 1. Convert to a NumPy array with its original float data type.
        float_array = np.array(pixel_list, dtype=np.float32)
        print(f"shape: {float_array.shape}")

        # 2. Find the minimum and maximum pixel values in the data.
        min_val = np.min(float_array)
        max_val = np.max(float_array)

        # 3. Avoid division by zero if all pixels are the same color.
        if max_val - min_val > 0:
            # Normalize the array to the range [0.0, 1.0].
            normalized_array = (float_array - min_val) / (max_val - min_val)
        else:
            # If all values are the same, the image is a single color.
            normalized_array = np.zeros_like(float_array)

        # 4. Scale the normalized data to the range [0, 255].
        scaled_array = normalized_array * 255

        # 5. Convert to an 8-bit integer array, which is what PIL expects.
        image_array = scaled_array.astype(np.uint8)
        # --- END OF NEW STEP ---

        # Create an image from the correctly scaled NumPy array.
        image = Image.fromarray(image_array)

        image.save(output_image_path)
        print(f"Image successfully normalized and saved to '{output_image_path}'")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- --- --- --- ---
# How to use the code
# --- --- --- --- ---
log_filename = '2M_loss_42_test.log'
output_filename = 'extracted_image_normalized.png'
save_normalized_log_image(log_filename, output_filename)