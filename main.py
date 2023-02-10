import sys
import scipy.io
import numpy as np
import torch
import pathlib
from PIL import Image
import kornia.geometry.transform
from model import Model


def main(full_limited_data=False):
    model_path = "model.pth"

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if len(sys.argv) > 3:
        print("The following command line arguments have been ignored:",
            ", ".join(sys.argv[3:]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

    paths = list(pathlib.Path(input_folder).glob("*.mat"))

    for path in paths:
        print("Processing", path, end="")

        # Load sinogram and angles
        ct_data = scipy.io.loadmat(str(path))["CtDataLimited"][0][0]
        sinogram = ct_data["sinogram"]
        angles = ct_data["parameters"]["angles"][0, 0][0]

        # Ensure that data is right
        assert len(angles) == sinogram.shape[0], "The number of angles must match the number of sinogram measurements"
        assert np.all(0.5 == np.diff(angles)), "Angles must be consecutive in 0.5° steps"
        assert np.all(0.0 <= angles), "Angles must be greater than 0°"
        assert np.all(angles <= 360), "Angles must be less or equal to 360°"
        assert len(angles) <= 181, "There must be no more than 181 angles"
        assert sinogram.shape[1] == 560, "The detector size of the sinogram must be 560"

        # Create input for neural network
        inputs = np.zeros((1, 2, 181, sinogram.shape[1]), dtype=np.float32)
        # Insert sinogram into input array
        inputs[0, 0, : sinogram.shape[0], :] = sinogram
        # Mark valid sinogram pixels
        inputs[0, 1, : sinogram.shape[0], :] = 1

        # Convert NumPy to PyTorch and upload data to GPU
        inputs = torch.tensor(inputs, device=device)

        with torch.no_grad():
            # Predict
            predictions = model(inputs)
            start_angle = torch.tensor(angles[:1], device=device).float()
            predictions = kornia.geometry.transform.rotate(predictions, start_angle)

            # Only process a single image at once for lower memory usage
            prediction = predictions[0, 0]

            # PyTorch to NumPy
            prediction = prediction.cpu().numpy()

            # Threshold prediction to get a binary mask
            prediction = prediction > prediction.mean()

            # Convert and write to file
            prediction = np.clip(prediction * 255, 0, 255).astype(np.uint8)
            prediction = Image.fromarray(prediction)
            prediction = prediction.convert("RGB")
            output_path = pathlib.Path(output_folder) / path.name.replace(".mat", ".png")
            prediction.save(output_path)

            print(" - writing to", output_path)


if __name__ == "__main__":
    main()
