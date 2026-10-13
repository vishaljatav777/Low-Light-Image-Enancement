"""Enhance one image or every image in a folder with a trained checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from low_light_enhancement.data.dataset import IMAGE_SUFFIXES
from low_light_enhancement.models import ZeroDCE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/enhanced"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not args.input.exists():
        parser.error(f"input not found: {args.input}")
    paths = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) if args.input.is_dir() else [args.input]
    paths = [p for p in paths if p.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        parser.error("input contains no supported images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroDCE().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                array = np.asarray(image, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
                enhanced, _ = model(tensor)
                output = enhanced.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                Image.fromarray((output * 255).round().astype(np.uint8)).save(args.output / f"{path.stem}_enhanced.png")
    print(f"Enhanced {len(paths)} image(s) into {args.output}")


if __name__ == "__main__":
    main()
