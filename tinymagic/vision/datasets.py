from glob import glob
from PIL import Image
from torch.utils.data import Dataset


class SegmetationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.transform = transform

        self.image_files = sorted(glob(f"{self.root}/Image/*"))
        self.mask_files = sorted(glob(f"{self.root}/Mask/*"))
        self.total_ims = len(self.image_files)
        self.total_gts = len(self.mask_files)

        assert self.total_ims == self.total_gts
        print(f"There are {self.total_ims} images and {self.total_gts} masks in the dataset")

    def __len__(self):
        return min(len(self.image_files), len(self.mask_files))

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # Load images using PIL
        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
