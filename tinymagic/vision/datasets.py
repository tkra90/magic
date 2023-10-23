import re
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


class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_length=1024, max_output_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.data)

    @staticmethod
    def generate_train_prompt(article: str, summary: str, prompt: str) -> str:
        txt = f"# Instruction: {prompt}\n"
        txt += f"# Input: {article}\n"
        txt += f"# Response: {summary}\n"
        return txt

    @staticmethod
    def text_cleaning(x):
        text = re.sub(r" .\n", ". ", x)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\^[^ ]+", "", text)
        text = re.sub(r"@[^\s]+", "", text)
        return text

    def __getitem__(self, index):
        article = self.text_cleaning(self.data.iloc[index]["article"])
        summary = self.text_cleaning(self.data.iloc[index]["highlights"])
        txt = self.generate_train_prompt(article, summary)
        txt = self.tokenizer(txt, truncation=True, max_length=self.max_input_length, padding="max_length")
        return {"article": article, "summary": summary, "text": txt}
