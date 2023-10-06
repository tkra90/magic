import matplotlib.pyplot as plt


def show_images_with_masks(dataset, indices=[0]):
    """
    Display a specified number of image-mask pairs from a Torch dataset.

    Args:
    - dataset: Torch dataset containing image-mask pairs (e.g., TensorDataset).
    - indices: list of image indices.
    """
    # Create a figure with multiple subplots
    img_number = len(indices)
    fig, axs = plt.subplots(img_number, 2, figsize=(7, 4 * img_number))

    for i, idx in enumerate(indices):
        img, mask = dataset[idx]
        img = img.permute(1, 2, 0).numpy()  # Convert to numpy image format (C, H, W) -> (H, W, C)
        if len(mask.shape) == 3:
            mask = mask.permute(1, 2, 0).numpy()
        else:
            mask = mask.numpy()

        # Display the image
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("Image")
        axs[i, 0].axis("off")

        # Display the mask
        axs[i, 1].imshow(mask)
        axs[i, 1].set_title("Mask")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
