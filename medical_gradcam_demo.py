"""Local chest X-ray classification + Grad-CAM demonstration.

Requirements (in a CUDA-enabled PyTorch environment):
    python -m pip install torchxrayvision grad-cam matplotlib scikit-image
    
Usage:
    python medical_gradcam_demo.py
    python medical_gradcam_demo.py --image path/to/xray.png --target "Cardiomegaly"
    
Outputs:
    results/medical_gradcam_demo.png
    results/classification_scores.csv
    
Research demonstration only: predictions are not clinical diagnoses and the example X-ray has no established ground-truth lesion segmentation here.
"""

import argparse
import csv
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch
import torchxrayvision as xrv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

SAMPLE_IMAGE_URL = (
    "https://huggingface.co/spaces/torchxrayvision/"
    "torchxrayvision-classifier/resolve/main/16747_3_1.jpg"
)

def load_xray(image_path: Path):
    # Return [1,1,224,2224] tensor and display RGB image [224,224,3]
    raw = skimage.io.imread(str(image_path))
    if raw.ndim == 3:
        raw = raw[..., :3].mean(axis=2)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got {raw.shape}")

    # This example assumes an 8-bit JPG/PNG input, not 16-bit DICOM data.
    normalized = xrv.datasets.normalize(raw.astype(np.float32), 255)
    normalized = normalized[None, :, :]
    normalized = xrv.datasets.XRayCenterCrop()(normalized)
    normalized = xrv.datasets.XRayResizer(224)(normalized)
    x = torch.from_numpy(np.ascontiguousarray(normalized)).unsqueeze(0).float()

    grayscale_01 = np.clip((normalized[0] + 1024.0) / 2048.0, 0, 1)
    display_rgb = np.repeat(grayscale_01[..., None], 3, axis=2).astype(np.float32)
    return x, display_rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("xray.jpg"))
    parser.add_argument("--target", default="Lung Opacity", help='Class to explain, e.g. "Lung Opacity" or "Cardiomegaly"')
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.image.exists() and args.image == Path("xray.jpg"):
        print("Downloading a public example X-ray...")
        urlretrieve(SAMPLE_IMAGE_URL, args.image)
    if not args.image.exists():
        parser.error(f"Image not found: {args.image}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: PyTorch cannot access CUDA; running on CPU.")

    x, display_rgb = load_xray(args.image)
    x = x.to(device)

    # Model weights are downloaded once and subsequently cached.
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(device).eval()
    pathologies = list(model.pathologies)
    if args.target not in pathologies:
        parser.error("Unknown target. Choose one of: " + ", ".join(pathologies))
    target_idx = pathologies.index(args.target)

    with torch.no_grad():
        scores = model(x)[0].cpu().numpy()
    ranking = sorted(zip(pathologies, scores), key=lambda item: -item[1])

    print("\nModel scores (not calibrated diagnostic probabilities):")
    for name, score in ranking:
        print(f" {name:29s} {score:.4f}")

    csv_path = args.out / "classification_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pathology", "model_score"])
        writer.writerows((name, float(score)) for name, score in ranking)

    # Hook the last dense block, before its final BatchNorm and inplace ReLU.
    # Grad-CAM computes gradients for the specifically selected pathology score.
    with GradCAM(model=model, target_layers=[model.features.denseblock4]) as cam:
        saliency = cam(
            input_tensor=x,
            targets=[ClassifierOutputTarget(target_idx)]
        )[0]

    overlay = show_cam_on_image(display_rgb, saliency, use_rgb=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    axes[0].imshow(display_rgb)
    axes[0].set_title("Input chest X-ray")
    axes[1].imshow(saliency, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM attribution")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay: {args.target}\nModel score: {float(scores[target_idx]):.3f}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("DenseNet121 - medical classification and Grad-CAM", fontsize=14)
    fig.tight_layout()
    fig_path = args.out / "medical_gradcam_demo.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_path}\nSaved: {csv_path}")

if __name__ == "__main__":
    main()