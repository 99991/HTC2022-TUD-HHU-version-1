import numpy as np
from PIL import Image
import scipy.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from numba import njit

def matthews_corrcoef(a, b):
    a, b = a.ravel(), b.ravel()
    (tp, fn), (fp, tn) = [a, 1.0 - a] @ np.array([b, 1.0 - b]).T
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator != 0 else 0

@njit("void(b1[:, :], b1[:, :], i8, i8)", cache=True)
def floodfill(is_valid, visited, x, y):
    w = is_valid.shape[0]
    h = is_valid.shape[1]
    queue = np.zeros((w * h, 2), dtype=np.int32)
    queue[0, 0] = x
    queue[0, 1] = y
    visited[y, x] = True
    n = 1
    i = 0
    while i < n:
        x = queue[i, 0]
        y = queue[i, 1]
        i += 1
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x2 = x + dx
                y2 = y + dy
                if 0 <= x2 < w and 0 <= y2 < h and is_valid[y2, x2] and not visited[y2, x2]:
                    queue[n, 0] = x2
                    queue[n, 1] = y2
                    visited[y2, x2] = True
                    n += 1

@njit("b1[:, :](b1[:, :])", cache=True)
def erode(image):
    w = image.shape[0]
    h = image.shape[1]
    result = np.zeros((h, w), dtype=image.dtype)
    for y in range(h):
        for x in range(w):
            all_pixels_in_window_are_set = True
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if abs(dx) + abs(dy) != 1: continue
                    x2 = x + dx
                    y2 = y + dy
                    if 0 <= x2 < w and 0 <= y2 < h and not image[y2, x2]:
                        all_pixels_in_window_are_set = False
            result[y, x] = all_pixels_in_window_are_set
    return result

def normalize(values):
    return (values - values.min()) / (values.max() - values.min())

print("level,     MCC,      PSNR,     SSIM")
for level in [1, 2, 3, 4, 5, 6, 7]:
    scores = []
    mses = []
    ssims = []
    psnrs = []

    for image in "abc":
        path = f"data/reconstructions/htc2022_0{level}{image}_limited.png"

        segmentation = np.array(Image.open(path).convert("L")) > 128

        ground_truth_segmentation_path = f"data/htc2022_test_data/htc2022_0{level}{image}_recon_fbp_seg.png"

        ground_truth_segmentation = np.array(Image.open(ground_truth_segmentation_path))

        is_outside = np.zeros(ground_truth_segmentation.shape, dtype=np.bool8)

        floodfill(ground_truth_segmentation == 0, visited=is_outside, x=0, y=0)

        is_outside = erode(is_outside)

        score = matthews_corrcoef(segmentation, ground_truth_segmentation)

        scores.append(score)

        path = f"data/reconstructions/htc2022_0{level}{image}_limited.npy"
        limited_angle = np.load(path)

        full_angle_fbp_path = f"data/htc2022_test_data/htc2022_0{level}{image}_recon_fbp.mat"
        full_angle_fbp = scipy.io.loadmat(str(full_angle_fbp_path))["reconFullFbp"]
        full_angle_fbp = normalize(full_angle_fbp)

        full_angle_fbp[is_outside] = 0

        mse = np.mean(np.square(limited_angle - full_angle_fbp))
        ssim = structural_similarity(full_angle_fbp, limited_angle, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
        psnr = peak_signal_noise_ratio(full_angle_fbp, limited_angle, data_range=1.0)

        mses.append(mse)
        ssims.append(ssim)
        psnrs.append(psnr)

    print(f"{level:5d}, {sum(scores):7.5f}, {np.mean(psnrs):9.6f}, {np.mean(ssims):.6f}")
print()
