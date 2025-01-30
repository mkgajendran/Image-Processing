import os
import shutil
import numpy as np
from PIL import Image, UnidentifiedImageError

def process_and_separate_image(image_path, output_folder):
    """
    1. Opens the image at image_path.
    2. Determines:
       - File format (png, jpeg, tiff, etc.)
       - Mode (RGB, RGBA, CMYK, L, etc.)
       - Bit depth (8, 16, 32, or float)
       - Whether an alpha channel is present and if it's 'empty' or in use.
    3. Prints all the above info and warnings if data will be lost on conversion.
    4. If it's grayscale, raises an error (no color channels to separate).
    5. Otherwise, converts or retains channels for R, G, B, and saves them.
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # 1. Open the image
    img = Image.open(image_path)
    # Basic file info
    file_name = os.path.basename(image_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    
    # 2. Determine file format, color mode, bit depth, alpha presence, etc.
    # --------------------------------------------------------------------
    # File format (based on extension or actual file info)
    if file_ext in [".jpg", ".jpeg"]:
        file_format = "JPEG"
    elif file_ext == ".png":
        file_format = "PNG"
    elif file_ext in [".tif", ".tiff"]:
        file_format = "TIFF"
    else:
        file_format = f"Unknown/Other ({file_ext})"

    # Color mode
    mode = img.mode  # e.g. "RGB", "RGBA", "L", "CMYK", "RGB;16", "I;16", etc.

    # Infer bit depth
    bit_depth = None
    # Check if mode has a suffix like ';16', ';32' which indicates high-bit
    if ";16" in mode:
        bit_depth = 16
    elif ";32" in mode:
        bit_depth = 32
    else:
        # Some modes are inherently 8 bits per channel (e.g., "RGB", "RGBA", "CMYK", "L").
        # But "I" can be 32-bit integer grayscale, "F" can be 32-bit float grayscale.
        if mode.startswith("I"):
            bit_depth = 32
        elif mode.startswith("F"):
            bit_depth = "float32"
        else:
            bit_depth = 8

    # Check alpha channel
    has_alpha = ("A" in mode.upper())
    alpha_status = "No alpha channel."
    if has_alpha:
        # Extract alpha channel (last band in .split())
        alpha_channel = img.split()[-1]
        alpha_arr = np.array(alpha_channel)
        amin, amax = alpha_arr.min(), alpha_arr.max()

        if amin == amax:
            # It's either all 0 or all max
            if amin == 0:
                alpha_status = "Alpha channel is fully transparent (empty)."
            else:
                # If 8-bit => 255, if 16-bit => 65535
                alpha_status = "Alpha channel is fully opaque (empty)."
        else:
            alpha_status = "Alpha channel is in use (varying transparency)."

    # Print discovered info
    print(f"Processing image: {file_name}")
    print(f"  - File format: {file_format}")
    print(f"  - Mode: {mode}")
    print(f"  - Bit depth (inferred): {bit_depth}")
    print(f"  - Alpha: {alpha_status}")

    # 3. Warn if data may be lost
    about_to_lose_data = False

    # a) If bit_depth > 8 and we must do .convert("RGB"), that’s a loss.
    if ((bit_depth == 16 or bit_depth == 32 or bit_depth == "float32") 
        and not mode.startswith("RGB")):
        about_to_lose_data = True
        print("WARNING: Converting from >8-bit to 8-bit will lose some information.")

    # b) If alpha channel is in use and we convert to RGB
    if has_alpha and "in use" in alpha_status:
        about_to_lose_data = True
        print("WARNING: Converting from RGBA/LA to RGB will discard the alpha channel.")

    # 4. If it's grayscale, raise error (no channels to separate)
    #    e.g., "L", "LA", "I", "I;16", "F"
    #    We define "no color channels" if there's only 1 channel or 1+alpha.
    if mode in ["L", "LA", "I", "I;16", "F"]:
        raise ValueError(f"ERROR: {file_name} is grayscale ({mode}). No R/G/B channels to separate.")

    # 5. Separate channels if it's color. We'll keep 16-bit if mode == "RGB;16"
    base_name = os.path.splitext(file_name)[0]

    if mode == "RGB;16":
        # Already 16-bit RGB, let’s split without dropping to 8 bits.
        arr = np.array(img, dtype=np.uint16)  # shape (H, W, 3)
        r_channel = arr[..., 0]
        g_channel = arr[..., 1]
        b_channel = arr[..., 2]

        r_img = Image.fromarray(r_channel, mode='I;16')
        g_img = Image.fromarray(g_channel, mode='I;16')
        b_img = Image.fromarray(b_channel, mode='I;16')

        r_img.save(os.path.join(output_folder, f"{base_name}_R_16bit.png"))
        g_img.save(os.path.join(output_folder, f"{base_name}_G_16bit.png"))
        b_img.save(os.path.join(output_folder, f"{base_name}_B_16bit.png"))

        print(f"  -> Saved 16-bit R/G/B channels for {file_name} in '{output_folder}'.")
    else:
        # Convert to standard 8-bit RGB
        img_rgb = img.convert("RGB")
        r_img, g_img, b_img = img_rgb.split()

        r_img.save(os.path.join(output_folder, f"{base_name}_R.png"))
        g_img.save(os.path.join(output_folder, f"{base_name}_G.png"))
        b_img.save(os.path.join(output_folder, f"{base_name}_B.png"))

        print(f"  -> Converted to 8-bit RGB and saved R/G/B channels for {file_name} in '{output_folder}'.")
    
    print("Done.\n")


def process_folder(folder_path):
    """
    For each valid image file in folder_path:
      1) Create a subfolder named after the file’s base name + file extension.
      2) Copy the original image into that subfolder.
      3) Process the image to separate channels into that same subfolder.
      4) If it's grayscale or corrupted, catch exceptions and print the error (skips channel separation).
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    for fname in os.listdir(folder_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_extensions:
            # Construct full path to the image
            full_path = os.path.join(folder_path, fname)
            base_name = os.path.splitext(fname)[0]

            # Create a unique subfolder by appending the extension name (without the dot),
            # to avoid collisions if there are multiple files named 'image.jpg' and 'image.png'.
            # e.g. "image.jpg" -> subfolder "image_jpg"
            subfolder_name = f"{base_name}_{ext[1:]}"  # remove '.' from extension
            output_subfolder = os.path.join(folder_path, subfolder_name)
            os.makedirs(output_subfolder, exist_ok=True)

            # Copy the original file into the subfolder
            dest_file_path = os.path.join(output_subfolder, fname)
            try:
                shutil.copy2(full_path, dest_file_path)
            except PermissionError as e:
                print(f"Permission error copying '{fname}': {e}")
                continue
            except OSError as e:
                print(f"OS error copying '{fname}': {e}")
                continue

            # Process and separate channels into that subfolder
            try:
                process_and_separate_image(full_path, output_subfolder)
            except ValueError as e:
                # For grayscale images or other explicit ValueErrors
                print(e)
            except (OSError, UnidentifiedImageError) as e:
                # For corrupted or unidentifiable files
                print(f"Could not open or process '{fname}': {e}")

if __name__ == "__main__":
    # Specify your folder path directly here:
    folder_path = r"/path/to/your/image/folder"
    
    if not os.path.isdir(folder_path):
        print(f"Invalid folder path: {folder_path}")
    else:
        process_folder(folder_path)
        print("All done.")
