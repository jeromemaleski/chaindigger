################
# This script runs the raw to rgb image conversion on a folder.
# Enter raw image directory path
# saved image path
# Replace width and height with the actual full dimensions of your raw data.
raw_directory_path_str = '/Users/jm/Data/20241030_cam1' # CHANGE THIS TO YOUR DIRECTORY PATH
image_width = 1440  # CHANGE THIS TO YOUR IMAGE WIDTH
image_height = 1080 # CHANGE THIS TO YOUR IMAGE HEIGHT
output_directory_path_str = '/Users/jm/Data/20241030_image' # CHANGE THIS TO YOUR DESIRED OUTPUT DIRECTORY

# Optional: Apply adaptive histogram equalization
histogram_eq = True # Set to False to disable

# Bayer pattern of the sensor, e.g., 'RGGB', 'GRBG', 'GBRG', 'BGGR'
# The colour-demosaicing library supports various patterns.
bayer_pattern = 'RGGB' 
# Demosaicing algorithm to use from colour-demosaicing library
# Options include: 'bilinear', 'malvar2004', 'menon2007'
demosaic_algorithm = 'bilinear'
################

#imports
import numpy as np
from pathlib import Path
from PIL import Image
# scikit-image for image processing
from skimage import exposure, util # util includes img_as_ubyte, exposure includes rescale_intensity
#import colour
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007)

DEMOSAICING_ALGORITHMS = {
        'bilinear': demosaicing_CFA_Bayer_bilinear,
        'malvar2004': demosaicing_CFA_Bayer_Malvar2004,
        'menon2007': demosaicing_CFA_Bayer_Menon2007,
    }

#functions

def get_raw_paths(directory):
    """
    Gets paths to all raw files from a directory path as pathlib.Path objects.
    """
    extensions = [".raw"]
    raw_paths = []
    directory_path = Path(directory) 

    if not directory_path.is_dir():
        print(f"Error: Directory not found at {directory}")
        return raw_paths

    try:
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                raw_paths.append(file_path) 
    except Exception as e:
        print(f"An error occurred while getting raw paths: {e}")
    return raw_paths

def raw_to_rgb_data(raw_filepath, width, height, target_bayer_pattern='RGGB', algorithm_name='bilinear'):
    """
    Reads a .raw file, unpacks 12-bit data, performs full demosaicing using colour-demosaicing,
    and returns a full-size RGB image array (uint16, 0-4095).
    """
    full_width = width
    full_height = height
    expected_size = int(full_width * full_height * 1.5)

    if not DEMOSAICING_ALGORITHMS:
        print("Critical Error: colour-demosaicing library not available. Cannot proceed.")
        return None
        
    demosaic_func = DEMOSAICING_ALGORITHMS.get(algorithm_name.lower())
    if not demosaic_func:
        print(f"Error: Demosaicing algorithm '{algorithm_name}' not found. Available: {list(DEMOSAICING_ALGORITHMS.keys())}")
        return None

    try:
        with open(Path(raw_filepath), 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)

        if raw_data.size != expected_size:
             print(f"Warning: File size mismatch for {raw_filepath.name}. Expected {expected_size}, got {raw_data.size}.")
             if raw_data.size < expected_size:
                 print(f"Error: File {raw_filepath.name} is smaller. Skipping.")
                 return None
             elif raw_data.size > expected_size:
                 print(f"Warning: File {raw_filepath.name} is larger. Truncating.")
                 raw_data = raw_data[:expected_size]

        if raw_data.size % 3 != 0:
            print(f"Warning: Raw data size ({raw_data.size}) for {raw_filepath.name} not multiple of 3. Truncating.")
            raw_data = raw_data[:-(raw_data.size % 3)]
            if raw_data.size == 0:
                print(f"Error: Raw data size 0 for {raw_filepath.name}. Skipping.")
                return None

        num_pixels = (raw_data.size // 3) * 2
        outval = np.zeros(num_pixels, dtype=np.uint16) 

        byte1 = raw_data[0::3].astype(np.uint16)
        byte2 = raw_data[1::3].astype(np.uint16)
        byte3 = raw_data[2::3].astype(np.uint16)
        
        min_len = min(len(byte1), len(byte2), len(byte3))
        byte1, byte2, byte3 = byte1[:min_len], byte2[:min_len], byte3[:min_len]

        outval[0:min_len*2:2] = byte1 + (np.bitwise_and(byte2, 0x0F) << 8)
        outval[1:min_len*2:2] = (byte3 << 4) + (np.bitwise_and(byte2, 0xF0) >> 4)
        
        if outval.size > full_width * full_height:
            outval = outval[:full_width * full_height]
        elif outval.size < full_width * full_height:
             print(f"Error: Decoded pixel count ({outval.size}) < expected ({full_width * full_height}) for {raw_filepath.name}. Skipping.")
             return None

        cfa_bayer_data_uint16 = outval.reshape((full_height, full_width))
        
        # Perform Demosaicing using colour-demosaicing library
        # The library functions typically expect float input [0,1] or integer input.
        # If integer, they scale it based on dtype. For uint16, max is 65535.
        # Our data is 12-bit (max 4095) stored in uint16.
        # To ensure correct scaling by the library if it assumes full uint16 range,
        # it's safer to convert to float and normalize to [0,1] based on 12-bit depth.
        
        cfa_bayer_data_float = cfa_bayer_data_uint16.astype(np.float32) / 4095.0
        cfa_bayer_data_float = np.clip(cfa_bayer_data_float, 0, 1) # Ensure it's strictly [0,1]

        print(f"Demosaicing with '{algorithm_name}' and pattern '{target_bayer_pattern}'...")
        rgb_image_float_0_1 = demosaic_func(cfa_bayer_data_float, pattern=target_bayer_pattern)
        # Output of demosaic_func is float, typically in range [0, 1]

        # Scale back to 0-4095 range for 12-bit depth
        rgb_image_scaled_float = rgb_image_float_0_1 * 4095.0
        
        # Clip to ensure values are within the valid 12-bit range and convert to uint16
        rgb_image_final_uint16 = np.clip(rgb_image_scaled_float, 0, 4095).astype(np.uint16)

        return rgb_image_final_uint16

    except FileNotFoundError:
        print(f"Error: File not found at {raw_filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while processing {raw_filepath.name}: {e}")
        import traceback
        traceback.print_exc()
        return None

#main script
def main():
    """Main function to execute the script."""
    raw_directory = Path(raw_directory_path_str)
    output_directory = Path(output_directory_path_str)

    # --- Input Validation ---
    if not raw_directory_path_str or raw_directory_path_str == 'path/to/your/raw/directory':
        print("Error: Please update 'raw_directory_path_str'.")
        return
    if not output_directory_path_str or output_directory_path_str == 'path/to/your/output/directory':
        print("Error: Please update 'output_directory_path_str'.")
        return
    if not isinstance(image_width, int) or image_width <= 0 or \
       not isinstance(image_height, int) or image_height <= 0:
        print(f"Error: 'image_width' and 'image_height' must be positive integers.")
        return
    if demosaic_algorithm.lower() not in DEMOSAICING_ALGORITHMS and DEMOSAICING_ALGORITHMS:
        print(f"Error: Demosaic_algorithm '{demosaic_algorithm}' unavailable. Choose from: {list(DEMOSAICING_ALGORITHMS.keys())}")
        return
    if not DEMOSAICING_ALGORITHMS :
         print("Error: colour-demosaicing library not loaded. Cannot proceed.")
         return

    if image_width % 2 != 0:
        print(f"Warning: 'image_width' ({image_width}) is odd.")
    if image_height % 2 != 0:
        print(f"Warning: 'image_height' ({image_height}) is odd.")

    try:
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {output_directory}")
    except Exception as e:
        print(f"Error creating output directory {output_directory}: {e}")
        return

    raw_file_paths = get_raw_paths(raw_directory)

    if not raw_file_paths:
        print(f"No .raw files found in {raw_directory}")
    else:
        print(f"Found {len(raw_file_paths)} .raw files.")
        processed_count = 0
        failed_count = 0
        for raw_filepath in raw_file_paths:
            output_filename = output_directory / f"{raw_filepath.stem}.jpg"
            print(f"Processing {raw_filepath.name}...")

            # Get demosaiced image (uint16, 0-4095)
            img_demosaiced_uint16  = raw_to_rgb_data(raw_filepath, image_width, image_height, 
                                                     target_bayer_pattern=bayer_pattern, 
                                                     algorithm_name=demosaic_algorithm)

            if img_demosaiced_uint16 is not None:
                print(f"Successfully demosaiced: {raw_filepath.name}. Shape: {img_demosaiced_uint16.shape}, dtype: {img_demosaiced_uint16.dtype}")

                # Convert to float [0,1] for further processing or direct saving
                # Input is 0-4095 uint16. rescale_intensity maps this to float 0-1.
                img_float_0_1 = exposure.rescale_intensity(
                    img_demosaiced_uint16, 
                    in_range=(0, 4095), 
                    out_range=(0.0, 1.0)
                ).astype(np.float32)

                if histogram_eq:
                    print(f"Applying adaptive histogram equalization to {raw_filepath.name}...")
                    # Input to equalize_adapthist should be float [0,1]
                    img_equalized_float_0_1 = exposure.equalize_adapthist(img_float_0_1, clip_limit=0.03)
                    # Output is float [0,1]
                    
                    # Convert the equalized [0,1] float image to 0-255 uint8 for saving
                    final_array_for_saving_uint8 = util.img_as_ubyte(img_equalized_float_0_1)
                    print(f"Applied histogram equalization.")
                else:
                    # Convert the non-equalized [0,1] float image to 0-255 uint8 for saving
                    final_array_for_saving_uint8 = util.img_as_ubyte(img_float_0_1)

                try:
                    if not final_array_for_saving_uint8.flags['C_CONTIGUOUS']:
                        final_array_for_saving_uint8 = np.ascontiguousarray(final_array_for_saving_uint8)
                    
                    img_to_save = Image.fromarray(final_array_for_saving_uint8, 'RGB')
                    img_to_save.save(output_filename)
                    print(f"Saved RGB image to {output_filename}")
                    processed_count += 1
                except Exception as e:
                    print(f"Error saving image {output_filename}: {e}")
                    failed_count += 1
            else:
                print(f"Failed to process {raw_filepath.name}.")
                failed_count += 1
        
        print(f"\nProcessing complete. Successfully processed: {processed_count}, Failed: {failed_count}")

if __name__ == '__main__':
    main()
