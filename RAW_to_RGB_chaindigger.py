
################
# This script runs the raw to rgb image conversion on a folder.
# does basic debayer wwith no interpolations
# Enter raw image directory path
# saved image path
# Replace width and height with the actual full dimensions of your raw data.
raw_directory_path_str = '/Users/jm/notebooks/chaindigger/20240905_image' # CHANGE THIS TO YOUR DIRECTORY PATH
image_width = 1440  # CHANGE THIS TO YOUR IMAGE WIDTH
image_height = 1080 # CHANGE THIS TO YOUR IMAGE HEIGHT
#for 1440x1080 image output is 8bit 540x720 RGB
output_directory_path_str = '/Users/jm/notebooks/chaindigger/20240905_out_basic' # CHANGE THIS TO YOUR DESIRED OUTPUT DIRECTORY
#use histogram equalizaion
histogram_eq=True #set to flase to disable
#
################

#imports

import numpy as np
from pathlib import Path
from PIL import Image
# scikit-image for histogram equalization
from skimage import exposure

#functions

def get_raw_paths(directory):
    """
    Gets paths to all raw files from a directory path as pathlib.Path objects.

    Args:
        directory (str or Path): The path to the directory.

    Returns:
        list: A list of pathlib.Path objects for the raw files found.
              Returns an empty list if the directory does not exist or no raw files are found.
    """
    extensions = [".raw"]
    raw_paths = []
    directory_path = Path(directory) # Ensure directory is a Path object

    if not directory_path.is_dir():
        print(f"Error: Directory not found at {directory}")
        return raw_paths

    try:
        # Use rglob to find files recursively with specified extensions
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                raw_paths.append(file_path) # Append the Path object

    except Exception as e:
        print(f"An error occurred while getting raw paths: {e}")

    return raw_paths


def raw_to_rgb_data(raw_filepath, width, height):
    """
    Reads a .raw file, extracts color channels, and composes them into a basic RGB image array.

    Args:
        raw_filepath (str or Path): The full path to the .raw file.
        width (int): The full horizontal size of the raw image data.
        height (int): The full vertical size of the raw image data.

    Returns:
        numpy.ndarray: A basic RGB image array uint16 12 bit (0-4095)
                       or None if an error occurs.
    """
    full_width = width
    full_height = height
    expected_size = full_width * full_height * 1.5 # Assuming 12-bit data packed into 3 bytes for every 2 pixels

    try:
        # Read the raw file as uint8 bytes
        # pathlib Path objects can be used directly with open()
        with open(Path(raw_filepath), 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)

        # Check if the file size matches the expected size
        if raw_data.size != expected_size:
             print(f"Warning: File size mismatch. Expected {expected_size} bytes, got {raw_data.size} bytes.")
             # Attempt to proceed, but be aware the data might be corrupted or format is different.
             # If the file is smaller than expected, truncate the data
             if raw_data.size < expected_size:
                 raw_data = raw_data[:expected_size]
                 print(f"Truncating raw data to {raw_data.size} bytes.")
             # If the file is larger, truncate to expected size
             elif raw_data.size > expected_size:
                 raw_data = raw_data[:expected_size]
                 print(f"Truncating raw data to {raw_data.size} bytes.")


        # Ensure raw_data size is a multiple of 3 for reshaping
        if raw_data.size % 3 != 0:
            print(f"Warning: Raw data size ({raw_data.size}) is not a multiple of 3 after size check. Truncating data.")
            raw_data = raw_data[:-(raw_data.size % 3)]


        # Reconstruct 12-bit values from 3 bytes (a, b, c) based on the MATLAB logic
        outval = np.zeros(full_width * full_height, dtype=np.uint16)

        # First values: out(1:3:end) + bitshift(bitand(out(2:3:end),15),8)
        # Python indexing: raw_data[0::3] + (raw_data[1::3] & 15) << 8
        outval[0::2] = raw_data[0::3].astype(np.uint16) + (np.bitwise_and(raw_data[1::3], 15).astype(np.uint16) << 8)

        # Second values: bitshift(out(3:3:end),4) + bitshift(bitand(out(2:3:end),240),-4)
        # Python indexing: (raw_data[2::3] << 4) + (np.bitwise_and(raw_data[1::3], 240).astype(np.uint16) >> 4)
        outval[1::2] = (raw_data[2::3].astype(np.uint16) << 4) + (np.bitwise_and(raw_data[1::3], 240).astype(np.uint16) >> 4)


        # Reshape the 12-bit values into the full image dimensions
        out_reshaped = outval.reshape((full_height, full_width))

        # Extract the four color channels based on the Bayer pattern
        # Assuming a RGGB or similar pattern where:
        # c1 = Top-Left (e.g., Red)
        # c2 = Top-Right (e.g., Green)
        # c3 = Bottom-Left (e.g., Green)
        # c4 = Bottom-Right (e.g., Blue)
        c1 = out_reshaped[0::2, 0::2] # Rows 0, 2, 4... and Columns 0, 2, 4...
        c2 = out_reshaped[0::2, 1::2] # Rows 0, 2, 4... and Columns 1, 3, 5...
        c3 = out_reshaped[1::2, 0::2] # Rows 1, 3, 5... and Columns 0, 2, 4...
        c4 = out_reshaped[1::2, 1::2] # Rows 1, 3, 5... and Columns 1, 3, 5...

        # Combine channels into a basic RGB image (simple averaging for green)
        # This is a simplified approach and not a full demosaicing algorithm.
        # The resulting image will have dimensions (height/2, width/2, 3)
        rgb_image_basic = np.zeros((height // 2, width // 2, 3), dtype=np.uint16)
        rgb_image_basic[:,:,0] = c1 # Red?
        rgb_image_basic[:,:,1] = (c2 + c3) // 2 # Green? (averaged)
        rgb_image_basic[:,:,2] = c4 # Blue?

        return rgb_image_basic

    except FileNotFoundError:
        print(f"Error: File not found at {raw_filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def normalize(arr):
    """Takes a numpy array and normalizeses and equalizes it 0-1,
    also scales it 0-255 and returns both versions"""
    
    # Normalize the array to the range 0-1
    arr_min = arr.min()
    arr_max = arr.max()
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    
    # Scale to the range 0-255
    scaled_arr = (normalized_arr * 255).astype(np.uint8)
    
    return normalized_arr,scaled_arr


#main script

# Use pathlib.Path for directory paths
raw_directory = Path(raw_directory_path_str)
output_directory = Path(output_directory_path_str)

# Create the output directory if it doesn't exist
if not output_directory.exists():
    output_directory.mkdir(parents=True, exist_ok=True) # Use mkdir with parents=True and exist_ok=True
    print(f"Created output directory: {output_directory}")


# Get all .raw file paths using the updated function
# This function now returns a list of Path objects
raw_file_paths = get_raw_paths(raw_directory)

if not raw_file_paths:
    print(f"No .raw files found in {raw_directory}")
else:
    print(f"Found {len(raw_file_paths)} .raw files.")
    for raw_filepath in raw_file_paths: # raw_filepath is now a Path object
        # Use pathlib's stem attribute to get the filename without extension
        output_filename = output_directory / f"{raw_filepath.stem}.jpg"

        print(f"Processing {raw_filepath.name}...") # Use .name attribute for filename

        # Call the raw_to_rgb_data function with the Path object
        im_array  = raw_to_rgb_data(raw_filepath, image_width, image_height)

        if im_array  is not None:
            print("Successfully read and processed raw data into RGB array.")
            print(f"Basic RGB image shape: {im_array .shape}")

            if histogram_eq is True:
                equalized = exposure.equalize_adapthist(im_array, clip_limit=0.03)
                array01,array255=normalize(equalized)
                print("Applied histogram normalization.")
            else:
                array01,array255=normalize(im_array)

            # Save the normalized RGB image using Pillow
            try:
                img_to_save = Image.fromarray(array255, 'RGB')
                # Use pathlib Path object for saving
                img_to_save.save(output_filename)
                print(f"Saved normalized RGB image to {output_filename}")
            except Exception as e:
                print(f"Error saving image: {e}")
                print("Please ensure you have Pillow installed (`pip install Pillow`)")

        else:
            print(f"Failed to process {raw_filepath.name}.")

