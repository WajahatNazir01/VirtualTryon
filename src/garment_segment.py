"""
garment_segment.py
Removes the background from a pasted clothing image using rembg (pretrained U^2-Net).
Produces a transparent PNG (RGBA) so it can be warped/overlaid later.
"""

import sys
from rembg import remove, new_session
from PIL import Image
import io

# u2net is ~170MB, much smaller than rembg's newer default model (bria-rmbg, ~1GB).
# Created once and reused so repeated calls don't reload the session.
_session = new_session("u2net")


def remove_background(input_path: str, output_path: str) -> str:
    """
    Loads an image, strips the background, saves an RGBA PNG.
    Returns the output path.
    """
    with open(input_path, "rb") as f:
        input_bytes = f.read()

    output_bytes = remove(input_bytes, session=_session)

    result = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    result.save(output_path)
    return output_path


if __name__ == "__main__":
    # Usage: python garment_segment.py path/to/shirt.jpg path/to/shirt_cutout.png
    if len(sys.argv) != 3:
        print("Usage: python garment_segment.py <input_image> <output_png>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    saved_to = remove_background(in_path, out_path)
    print(f"Saved transparent garment cutout to: {saved_to}")