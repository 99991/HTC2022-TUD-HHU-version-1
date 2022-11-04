import scipy.io
import numpy as np
from PIL import Image
from pathlib import Path


def main(full_limited_data=False):
    for src_path in Path("data/limited").glob("*.mat"):
        dst_path = Path("data/limited_png") / src_path.name.replace(".mat", ".png")
        print("Convert", src_path, "to", dst_path)

        # Load sinogram
        ct_data = scipy.io.loadmat(str(src_path))["CtDataLimited"][0][0]
        limited_sinogram = ct_data["sinogram"]
        angles = ct_data["parameters"]["angles"][0, 0][0]

        start = round(2 * angles[0])

        sinogram = np.zeros((721, 560))
        sinogram[start:start + len(angles)] = limited_sinogram

        # Save as PNG
        sinogram -= sinogram.min()
        sinogram /= sinogram.max()
        sinogram = np.clip(sinogram * 255, 0, 255).astype(np.uint8)
        Image.fromarray(sinogram).convert("L").save(str(dst_path))

if __name__ == "__main__":
    main()
