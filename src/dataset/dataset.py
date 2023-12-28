import os

import cv2
import pandas as pd

from torch.utils.data import Dataset


class CLIPDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        tokenizer,
        transform,
        txt_max_length: int = 200,
    ):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.txt_max_length = txt_max_length

        # dataframe
        self.data = df

        # encoded_captions
        captions = self.data["caption"].tolist()
        self.encoded_captions = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=self.txt_max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]["image_name"]
        caption = self.data.iloc[idx]["caption"]
        game_name = self.data.iloc[idx]["label_game"]
        genre_name = self.data.iloc[idx]["label_genre"]
        game_label = self.data.iloc[idx]["game_class"]
        genre_label = self.data.iloc[idx]["genre_class"]
        image_path = self.data.iloc[idx]["image_path"]

        # txt prep
        item = {k: v[idx] for k, v in self.encoded_captions.items()}

        # img prep
        img_path = os.path.join(self.img_dir, image_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]

        item["image"] = img
        item["game_label"] = game_label
        item["genre_label"] = genre_label

        item["caption"] = caption
        item["game_name"] = game_name
        item["genre_name"] = genre_name

        return item
