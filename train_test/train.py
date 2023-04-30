import os
import sys
import time
import kornia
import numpy as np
import torch
from PIL import Image
from forward_operator_astra import ForwardOperator
sys.path.append("..")
from model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

forward_operator = ForwardOperator()

def make_batch(full_images, sinogram_cache, indices):
    max_angles = 181
    sinogram_scale = 2.25 / 84900
    noise_std = 1000.0

    batch_size = indices.shape[0]
    images = full_images[indices]

    # Create sinograms
    sinograms = []
    for image, index in zip(images, indices):
        if sinogram_cache is not None and index not in sinogram_cache:
            sinogram_cache[index] = forward_operator(image)
        sinograms.append(sinogram_cache[index])
    sinograms = np.array(sinograms)

    # Make random number of sinogram angles
    p = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    p /= np.sum(p)
    num_angles = np.random.choice([61, 81, 101, 121, 141, 161, 181], batch_size, p=p)

    # Make random start angle
    start_angle_indices = np.random.randint(sinograms.shape[1] - num_angles + 1, size=batch_size)

    # Subsample sinograms by taking sinogram[start_angle_index:start_angle_index + num_angles]
    inputs = np.zeros((batch_size, 2, max_angles, sinograms.shape[2]), dtype=np.float32)
    for i in range(batch_size):
        inputs[i, 0, :num_angles[i]] = sinogram_scale * sinograms[i, start_angle_indices[i]:start_angle_indices[i] + num_angles[i]]
        inputs[i, 1, :num_angles[i]] = 1

    # NumPy arrays to Torch tensors
    inputs = torch.from_numpy(inputs).to(device)
    # (n, h, w) -> (n, 1, h, w)
    targets = torch.from_numpy(images).to(device).unsqueeze(1).float() / 255.0
    start_angle_indices = torch.from_numpy(start_angle_indices).to(device)

    # Add noise to sinograms, but only where mask is valid
    with torch.no_grad():
        inputs[:, 0, :] += sinogram_scale * noise_std * inputs[:, 1, :] * torch.randn(inputs[:, 0, :].shape, device=device)

    return inputs, targets, start_angle_indices

def main():
    np.random.seed(0)
    torch.manual_seed(0)

    tag = "train-baseline"
    num_epochs = 100
    lr = 1e-4
    num_workers = 16
    images_path = "images.npy"
    # Decrease batch size if you run out of VRAM
    # Could implement gradient accumulation for smaller GPUs
    batch_size = 4
    # Choose mmap_mode "r" if you do not have enough RAM
    mmap_mode = "r"
    mmap_mode = None
    # Set sinogram_cache to None if you do not have enough RAM
    sinogram_cache = None
    sinogram_cache = {}

    full_images = np.load(images_path, mmap_mode=mmap_mode)

    print("Creating model and optimizer...")
    model = Model()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    def rotate(x, start_angle_indices):
        return kornia.geometry.transform.rotate(x, start_angle_indices.float() * 0.5)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Training for {num_epochs} epochs")
    print()

    log_dir = f"log/{time.strftime('%Y.%m.%d-%H.%M.%S')}-{tag}"
    os.makedirs(log_dir)

    start_time = time.perf_counter()
    all_indices = np.arange(len(full_images))
    for epoch in range(1, 1 + num_epochs):
        model.train()
        train_losses = []

        num_batches = len(full_images) // batch_size
        for i_batch in range(num_batches):
            batch_start_time = time.perf_counter()
            indices = all_indices[i_batch * batch_size: (i_batch + 1) * batch_size]
            inputs, targets, start_angle_indices = make_batch(full_images, sinogram_cache, indices)
            out = model(inputs)
            predictions = rotate(out, start_angle_indices)
            train_loss = torch.mean(torch.square(predictions - targets))
            optim.zero_grad()
            train_loss.backward()
            optim.step()
            train_losses.append(train_loss.detach().item())
            # Print progress more often at beginning because more things can go wrong there
            if i_batch in [1, 2, 3, 4, 5, 10, 50, 100, 500, num_batches] or i_batch % 1000 == 1:
                done = (epoch - 1) * num_batches + i_batch
                batch_elapsed_time = time.perf_counter() - batch_start_time
                elapsed_time = time.perf_counter() - start_time
                remaining_time = batch_elapsed_time * (num_epochs * num_batches - done)
                print(f"{epoch}/{num_epochs} - {i_batch + 1:05d}/{num_batches} - {train_loss:.6f} - {elapsed_time:8.0f} sec, {remaining_time:8.0f} sec remaining ({remaining_time/3600:6.3f} hrs) - {time.strftime('%Y.%m.%d-%H.%M.%S')}")

                with torch.no_grad():
                    grid = make_grid([
                        inputs[:, :1, :, :],
                        targets,
                        predictions,
                    ])
                    Image.fromarray(grid).convert("RGB").save(f"{log_dir}/epoch_{epoch}_{i_batch}.jpg")

        # Shuffle dataset
        np.random.shuffle(all_indices)

        torch.save(model.state_dict(), f"{log_dir}/model_{epoch}.pth")
        torch.save(model.state_dict(), f"{log_dir}/model_latest.pth")
        torch.save(optim.state_dict(), f"{log_dir}/optim_latest.pth")
        print(f"Saved {epoch}")

def img2uint8(img):
    assert img.dtype in [np.uint8, np.float32, np.float64]

    if img.dtype in [np.float32, np.float64]:
        img_min = img.min()
        img_max = img.max()
        if img_min != img_max:
            img = (img - img_min) / (img_max - img_min)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img


def img2rgba(img):
    alpha_value = {
        np.dtype("uint8"): 255,
        np.dtype("float32"): 1.0,
        np.dtype("float64"): 1.0,
    }[img.dtype]

    if len(img.shape) == 2:
        alpha = np.full(img.shape, alpha_value, img.dtype)
        return np.stack([img, img, img, alpha], axis=2)

    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            return img

        alpha = np.full([img.shape[0], img.shape[1], 1], alpha_value, img.dtype)

        if img.shape[2] == 1:
            return np.concatenate([img, img, img, alpha], axis=2)
        elif img.shape[2] == 3:
            return np.concatenate([img, alpha], axis=2)
        else:
            raise ValueError(f"img.shape[2] should be in [1, 3, 4], but is {img.shape[2]} instead")
    else:
        raise ValueError(f"len(img.shape) should be in [2, 3], but is {len(img.shape)} instead")


def to_numpy(img):
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Assume torch.tensor
    elif not isinstance(img, np.ndarray):
        img = img.detach().cpu().numpy()


    # Move color axis (probably smallest dimension) to back for HWC format
    if len(img.shape) == 3:
        color_axis = np.argmin(img.shape)
        img = np.moveaxis(img, color_axis, 2)

    return img


def make_grid(img_rows, center_x=True, center_y=True, pad=1, bg_color=(200, 220, 240, 255)):
    img_rows = [[img2rgba(img2uint8(to_numpy(img)))
        for img in row] for row in img_rows]

    ny = len(img_rows)
    nx = max(len(row) for row in img_rows)

    ws = [0] * nx
    hs = [0] * ny

    for iy, row in enumerate(img_rows):
        for ix, img in enumerate(row):
            ws[ix] = max(ws[ix], img.shape[1])
            hs[iy] = max(hs[iy], img.shape[0])

    ws = [w + 2 * pad for w in ws]
    hs = [h + 2 * pad for h in hs]

    xs = np.cumsum([0] + ws)
    ys = np.cumsum([0] + hs)

    grid = np.zeros((ys[-1], xs[-1], 4), dtype=np.uint8)
    grid[:, :, :] = bg_color

    for iy, row in enumerate(img_rows):
        for ix, img in enumerate(row):

            ih, iw = img.shape[:2]

            x = xs[ix]
            y = ys[iy]

            if center_x:
                x += (ws[ix] - iw) // 2

            if center_y:
                y += (hs[iy] - ih) // 2

            grid[y:y+ih, x:x+iw] = img

    return grid

main()
