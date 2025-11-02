import os
import time
import numpy as np
import matplotlib.pyplot as plt
from preprocess import PreProcess
from ToTensor import ToTensorBatch
from DataLoader import DataLoader


def main():
    data_dir = r"C:\Users\eitan\Downloads\rsna-ihd-dataset\rsna-intracranial-hemorrhage-detection\stage_2_train"
    csv_path = r"C:\Users\eitan\Downloads\rsna-ihd-dataset\rsna-intracranial-hemorrhage-detection\stage_2_train.csv"

    pre = PreProcess(input_dir=data_dir, csv_path=csv_path)
    to_tensor = ToTensorBatch()
    loader = DataLoader(data_dir=data_dir, csv_path=csv_path, batch_size=4,preprocessor=pre, to_tensor=to_tensor, shuffle=True)

    print(f"Loaded {len(loader.file_paths)} images.")
    print(f"Number of batches per epoch: {len(loader)}")

    test_file = loader.file_paths[0]
    print("\nProcessing single image:")
    start = time.time()
    img = pre.preprocess(test_file)
    single_time = time.time() - start
    print(f"Single image processed in {single_time:.3f} sec, shape: {img.shape}")

    plt.imshow(img.squeeze(), cmap="gray")
    plt.title("Single image - PreProcess output")
    plt.axis("off")
    plt.show()

    print("\nLoading one batch from DataLoader...")
    start = time.time()
    x_batch, y_batch = loader[0]
    batch_time = time.time() - start

    print(f"Batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")
    print(f"Batch loaded in {batch_time:.3f} sec")
    print("Example label vector:", y_batch[0])

    fig, axes = plt.subplots(1, min(4, x_batch.shape[0]), figsize=(15, 5))
    hemorrhage_types = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]

    for i in range(min(4, x_batch.shape[0])):
        axes[i].imshow(x_batch[i].squeeze(), cmap="gray")
        labels_str = ", ".join([t for t, v in zip(hemorrhage_types, y_batch[i]) if v == 1])
        if labels_str == "":
            labels_str = "No hemorrhage"
        axes[i].set_title(labels_str, fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    print("\nPipeline test completed successfully.")


if __name__ == "__main__":
    main()
