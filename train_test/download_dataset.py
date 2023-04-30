import requests
import zipfile
import os
import io
import shutil

url = "https://zenodo.org/record/7418878/files/htc2022_test_data.zip"

if not os.path.isdir("data/htc2022_test_data"):
    print(f"Downloading {url} to data/htc2022_test_data")
    with requests.get(url) as response:
        assert response.status_code == 200, f"Request to {url} failed: {response.reason}"

        os.makedirs("data/htc2022_test_data")

        zip_data = io.BytesIO(response.content)

        # Open the zip file
        with zipfile.ZipFile(zip_data) as zip_file:
            # Extract the zip file to the target directory
            zip_file.extractall("data")

    print(f"Downloaded {url} and extracted content to data/htc2022_test_data")

if not os.path.isdir("data/htc2022_test_data_limited"):
    os.makedirs("data/htc2022_test_data_limited")

    for level in [1, 2, 3, 4, 5, 6, 7]:
        for name in "abc":
            src = f"data/htc2022_test_data/htc2022_0{level}{name}_limited.mat"
            dst = f"data/htc2022_test_data_limited/htc2022_0{level}{name}_limited.mat"
            shutil.copy(src, dst)
