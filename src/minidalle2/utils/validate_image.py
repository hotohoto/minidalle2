from pathlib import Path

import PIL


def validate_image(image_path: Path):
    if image_path.exists():
        try:
            PIL.Image.open(image_path)
            return True
        except PIL.UnidentifiedImageError:
            image_path.unlink()
    return False
