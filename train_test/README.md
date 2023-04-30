# Training and testing

1. In this directory, run `python3 generate_dataset.py [number of images]` to generate a dataset.
2. Run `python3 train.py` and wait a day or two.
3. Download the test dataset with `python3 download_dataset.py`
4. Switch to the parent directory and overwrite `model.pth` with the latest model from `train_test/log/<your training run>/model_latest.pth`.
5. In the parent directory, run `python3 main.py train_test/data/htc2022_test_data_limited train_test/data/reconstructions`
6. In `train_test` directory, run `python3 test.py` to compute scores for images in `train_test/data/reconstructions` directory.
