"""Train Zero-DCE on an unpaired folder of low-light images."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from low_light_enhancement.data import LowLightDataset
from low_light_enhancement.losses import ZeroDCELoss
from low_light_enhancement.models import ZeroDCE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/weights"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 16:
        parser.error("epochs and batch size must be positive; image size must be at least 16")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LowLightDataset(args.data, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    model, criterion = ZeroDCE().to(device), ZeroDCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images in loader:
            images = images.to(device)
            enhanced, curves = model(images)
            loss = criterion(images, enhanced, curves)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        checkpoint = args.output / f"zerodce_epoch_{epoch + 1:03d}.pth"
        torch.save({"model": model.state_dict(), "epoch": epoch + 1}, checkpoint)
        print(f"Epoch {epoch + 1}/{args.epochs} | loss {total / len(loader):.5f} | saved {checkpoint}")


if __name__ == "__main__":
    main()
