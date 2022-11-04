from main_confMatrix import calcScore as calc_score
from pathlib import Path
import numpy as np
import sys


def main():
    directory = sys.argv[1] if len(sys.argv) == 2 else "output"

    scores = []
    for predicted_segmentation in Path(directory).glob("*.png"):
        which = predicted_segmentation.name.split("_")[0]

        ground_truth_segmentation = f"data/htc2022_{which}_full_recon_fbp_seg.png"

        score = calc_score(str(predicted_segmentation), ground_truth_segmentation)

        print(score, predicted_segmentation, ground_truth_segmentation)

        scores.append(score)

    standard_error = np.std(scores) / np.sqrt(len(scores))

    print(f"\nMean score {np.mean(scores):.5f} +- {standard_error:.5f} standard error")

if __name__ == "__main__":
    main()
