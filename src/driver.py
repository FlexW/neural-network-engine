#!/usr/bin/env python3
"""
Neural Network Inference Engine Driver

A flexible driver that wraps the C inference engine, handles image preprocessing,
and displays results based on model family.

Usage:
    python driver.py -config <config_file> -input <image_file> -family <model_family>

Examples:
    python driver.py -config model.cfg -input inputs/image_0.ubyte -family mnist
    python driver.py -config mobilenetv2.cfg -input image.png -family mobilenetv2 -classes imagenet_classes.txt
"""

import argparse
import re
import signal
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


def _sigint_handler(signum, frame):
    """Handle Ctrl+C by exiting immediately."""
    sys.exit(0)


signal.signal(signal.SIGINT, _sigint_handler)

import numpy as np
from PIL import Image


def parse_config(config_path: str) -> dict:
    """Parse the custom configuration file format."""
    with open(config_path, "r") as f:
        content = f.read()

    config = {}

    # Extract model_path
    match = re.search(r'model_path:\s*"([^"]+)"', content)
    if match:
        config["model_path"] = match.group(1)

    # Extract batch_size
    match = re.search(r"batch_size:\s*(\d+)", content)
    if match:
        config["batch_size"] = int(match.group(1))
    else:
        config["batch_size"] = 1

    # Extract normalize_input
    match = re.search(r"normalize_input:\s*(\d+)", content)
    if match:
        config["normalize_input"] = int(match.group(1)) != 0
    else:
        config["normalize_input"] = False

    # Extract inputs array
    config["inputs"] = parse_inout_array(content, "inputs")

    # Extract outputs array
    config["outputs"] = parse_inout_array(content, "outputs")

    return config


def parse_inout_array(content: str, array_name: str) -> list:
    """Parse inputs or outputs array from config content."""
    pattern = rf"{array_name}:\s*\[(.*?)\](?=\s*(?:inputs:|outputs:|$))"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return []

    array_content = match.group(1)
    entries = []

    object_pattern = r"\{([^}]+)\}"
    for obj_match in re.finditer(object_pattern, array_content):
        obj_content = obj_match.group(1)
        entry = {}

        name_match = re.search(r'name:\s*"([^"]+)"', obj_content)
        if name_match:
            entry["name"] = name_match.group(1)

        dtype_match = re.search(r'data_type:\s*"([^"]+)"', obj_content)
        if dtype_match:
            entry["data_type"] = dtype_match.group(1)

        shape_match = re.search(r"shape:\s*\[([^\]]+)\]", obj_content)
        if shape_match:
            shape_str = shape_match.group(1).strip()
            entry["shape"] = [int(x) for x in shape_str.split()]

        entries.append(entry)

    return entries


def parse_tensor_output(stdout: str) -> np.ndarray:
    """Extract tensor values from inference engine output."""
    lines = stdout.strip().split("\n")

    in_tensor = False
    values = []

    for line in lines:
        line = line.strip()
        if line == "TENSOR_OUTPUT_START":
            in_tensor = True
            continue
        elif line == "TENSOR_OUTPUT_END":
            break
        elif in_tensor and line:
            try:
                values.append(float(line))
            except ValueError:
                continue

    if not values:
        raise ValueError("No tensor values found in output")

    return np.array(values, dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for the array."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


class ImageLoader:
    """Load and convert various image formats."""

    def __init__(self, config: dict):
        self.config = config
        self.input_shape = config["inputs"][0]["shape"] if config["inputs"] else None
        self._temp_file = None

    def load(self, image_path: str) -> tuple[str, np.ndarray]:
        """
        Load an image and return (path_to_use, image_data).

        For ubyte/raw files, returns the original path.
        For PNG/JPEG, converts to raw format and returns temp file path.
        """
        path = Path(image_path)
        suffix = path.suffix.lower()

        if suffix == ".ubyte":
            return self._load_ubyte(image_path)
        elif suffix == ".raw":
            return self._load_raw(image_path)
        elif suffix in [".png", ".jpg", ".jpeg"]:
            return self._load_and_convert_image(image_path)
        else:
            raise ValueError(f"Unsupported image format: {suffix}")

    def _load_ubyte(self, image_path: str) -> tuple[str, np.ndarray]:
        """Load a .ubyte file (MNIST format)."""
        with open(image_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)

        if len(data) == 784:
            data = data.reshape(28, 28)

        return image_path, data

    def _load_raw(self, image_path: str) -> tuple[str, np.ndarray]:
        """Load a .raw file (CHW binary format)."""
        with open(image_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)

        if self.input_shape:
            # Shape is typically [batch, channels, height, width]
            shape = (
                self.input_shape[1:] if self.input_shape[0] == 1 else self.input_shape
            )
            data = data.reshape(shape)

        return image_path, data

    def _load_and_convert_image(self, image_path: str) -> tuple[str, np.ndarray]:
        """Load PNG/JPEG, convert to raw format, save to temp file."""
        if not self.input_shape:
            raise ValueError("Input shape required for image conversion")

        # Parse shape: [batch, channels, height, width]
        _, channels, height, width = self.input_shape

        img = Image.open(image_path)

        # Convert to appropriate mode
        if channels == 1:
            img = img.convert("L")
        elif channels == 3:
            img = img.convert("RGB")
        elif channels == 4:
            img = img.convert("RGBA")

        # Resize (use Resampling.LANCZOS for Pillow >= 10, fallback to LANCZOS)
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img = img.resize((width, height), resample)

        # Convert to numpy array
        data = np.array(img, dtype=np.uint8)

        # Convert HWC -> CHW for multi-channel images
        if channels > 1:
            data = np.transpose(data, (2, 0, 1))

        # Save to temp file
        self._temp_file = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
        data.tofile(self._temp_file.name)

        return self._temp_file.name, data

    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_file:
            try:
                Path(self._temp_file.name).unlink()
            except Exception:
                pass


class InferenceRunner:
    """Invoke the C inference engine as a subprocess."""

    def __init__(self, engine_path: str = ".out/neural_net_engine"):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            # Try relative to script location
            script_dir = Path(__file__).parent
            self.engine_path = script_dir / engine_path

    def run(
        self, config_path: str, input_path: str, verbose: bool = False
    ) -> tuple[np.ndarray, float]:
        """Run inference and return output tensor with elapsed time."""
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Inference engine not found: {self.engine_path}")

        cmd = [str(self.engine_path), "-config", config_path, "-input", input_path]

        if verbose:
            print(f"Running: {' '.join(cmd)}")

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_time = time.time() - start_time

        if result.returncode != 0:
            print(f"Error running inference engine:")
            print(result.stderr)
            raise RuntimeError(f"Inference engine failed with code {result.returncode}")

        if verbose:
            print("Engine stdout:")
            print(result.stdout)

        return parse_tensor_output(result.stdout), elapsed_time


class DisplayHandler(ABC):
    """Base class for model-specific display logic."""

    @abstractmethod
    def display(
        self,
        image_data: np.ndarray,
        output_tensor: np.ndarray,
        classes_path: Optional[str] = None,
        output_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Display the inference result."""
        pass


class MnistDisplayHandler(DisplayHandler):
    """Display handler for MNIST digit classification."""

    def display(
        self,
        image_data: np.ndarray,
        output_tensor: np.ndarray,
        classes_path: Optional[str] = None,
        output_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Display MNIST result with detected digit."""
        import matplotlib.pyplot as plt

        # Apply softmax to get probabilities
        probabilities = softmax(output_tensor)
        predicted_digit = np.argmax(probabilities)
        confidence = probabilities[predicted_digit]

        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

        if verbose:
            print("\nAll probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"  {i}: {prob:.4f} ({prob*100:.2f}%)")

        # Reshape image for display
        if image_data.ndim == 1:
            display_img = image_data.reshape(28, 28)
        elif image_data.ndim == 3:
            display_img = image_data.squeeze()
        else:
            display_img = image_data

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(display_img, cmap="gray")
        ax.set_title(
            f"Detected: {predicted_digit} (confidence: {confidence:.2%})", fontsize=14
        )
        ax.axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Result saved to: {output_path}")
        else:
            plt.show()


class MobileNetV2DisplayHandler(DisplayHandler):
    """Display handler for MobileNetV2 ImageNet classification."""

    def display(
        self,
        image_data: np.ndarray,
        output_tensor: np.ndarray,
        classes_path: Optional[str] = None,
        output_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Display MobileNetV2 result with class name."""
        import matplotlib.pyplot as plt

        # Load class labels
        labels = None
        if classes_path and Path(classes_path).exists():
            with open(classes_path) as f:
                labels = [line.strip() for line in f.readlines()]

        # Apply softmax to get probabilities
        probabilities = softmax(output_tensor)
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

        # Get class name
        if labels and predicted_class < len(labels):
            class_name = labels[predicted_class]
        else:
            class_name = f"Class {predicted_class}"

        print(f"Predicted: {class_name}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")

        # Show top-5 predictions
        top_k = min(5, len(probabilities))
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        print(f"\nTop-{top_k} predictions:")
        for i, idx in enumerate(top_indices):
            if labels and idx < len(labels):
                print(
                    f"  {i+1}. {labels[idx]}: {probabilities[idx]:.4f} ({probabilities[idx]*100:.2f}%)"
                )
            else:
                print(
                    f"  {i+1}. Class {idx}: {probabilities[idx]:.4f} ({probabilities[idx]*100:.2f}%)"
                )

        # Reshape image for display (CHW -> HWC)
        if image_data.ndim == 3 and image_data.shape[0] in [1, 3, 4]:
            display_img = np.transpose(image_data, (1, 2, 0))
        else:
            display_img = image_data

        # Normalize for display if needed
        if display_img.max() > 1:
            display_img = display_img / 255.0

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(display_img)
        ax.set_title(f"{class_name}\n(confidence: {confidence:.2%})", fontsize=14)
        ax.axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"\nResult saved to: {output_path}")
        else:
            plt.show()


class Yolo11nDisplayHandler(DisplayHandler):
    """Display handler for YOLO11n object detection (placeholder)."""

    def display(
        self,
        image_data: np.ndarray,
        output_tensor: np.ndarray,
        classes_path: Optional[str] = None,
        output_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Display YOLO11n result with bounding boxes."""
        raise NotImplementedError(
            "YOLO11n display handler is not yet implemented. "
            "This will draw bounding boxes on the input image."
        )


def get_display_handler(family: str) -> DisplayHandler:
    """Factory function to get the appropriate display handler."""
    handlers = {
        "mnist": MnistDisplayHandler,
        "mobilenetv2": MobileNetV2DisplayHandler,
        "yolo11n": Yolo11nDisplayHandler,
    }

    if family not in handlers:
        raise ValueError(
            f"Unknown model family: {family}. Supported: {list(handlers.keys())}"
        )

    return handlers[family]()


def main():
    parser = argparse.ArgumentParser(
        description="Inference Engine Driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python driver.py -config model.cfg -input inputs/image_0.ubyte -family mnist
  python driver.py -config mobilenetv2.cfg -input image.png -family mobilenetv2 \\
      -classes mobilenetv2_raw_images/imagenet_classes.txt
        """,
    )

    parser.add_argument(
        "-config",
        "--config",
        required=True,
        help="Path to model configuration file (.cfg)",
    )
    parser.add_argument(
        "-input", "--input", required=True, help="Input image: ubyte, raw, png, jpeg"
    )
    parser.add_argument(
        "-family",
        "--family",
        required=True,
        choices=["mnist", "mobilenetv2", "yolo11n"],
        help="Model family: mnist, mobilenetv2, yolo11n",
    )
    parser.add_argument(
        "-classes",
        "--classes",
        help="Path to classes text file (e.g., imagenet_classes.txt)",
    )
    parser.add_argument(
        "-o", "--output", help="Output file path for saving result image"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.classes and not Path(args.classes).exists():
        print(f"Error: Classes file not found: {args.classes}")
        sys.exit(1)

    # Parse config
    config = parse_config(args.config)

    if args.verbose:
        print(f"Configuration loaded from: {args.config}")
        print(f"  Model: {config.get('model_path', 'N/A')}")
        print(
            f"  Input shape: {config['inputs'][0]['shape'] if config['inputs'] else 'N/A'}"
        )
        print(f"  Normalize input: {config.get('normalize_input', False)}")
        print()

    # Load image
    image_loader = ImageLoader(config)
    try:
        input_path, image_data = image_loader.load(args.input)

        if args.verbose:
            print(f"Image loaded: {args.input}")
            if input_path != args.input:
                print(f"  Converted to: {input_path}")
            print(f"  Shape: {image_data.shape}")
            print()

        # Run inference
        runner = InferenceRunner()
        output_tensor, inference_time = runner.run(
            args.config, input_path, verbose=args.verbose
        )
        print(f"Inference time: {inference_time * 1000:.2f}ms")

        if args.verbose:
            print(f"Output tensor shape: {output_tensor.shape}")
            print(
                f"Output range: [{output_tensor.min():.4f}, {output_tensor.max():.4f}]"
            )
            print()

        # Display result
        handler = get_display_handler(args.family)
        handler.display(
            image_data=image_data,
            output_tensor=output_tensor,
            classes_path=args.classes,
            output_path=args.output,
            verbose=args.verbose,
        )

    finally:
        image_loader.cleanup()


if __name__ == "__main__":
    main()
