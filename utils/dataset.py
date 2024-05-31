from torch.utils.data import Dataset
from PIL import Image

import os


class HazeDataset(Dataset):
    def __init__(self, haze_dir, image_dir=None, transform=None):
        self.haze_dir = haze_dir
        self.image_dir = image_dir

        self.transform = transform

        self.images = []
        self.pair_images()

    def pair_images(self):
        if self.image_dir is None:
            for haze in os.listdir(self.haze_dir):
                self.images.append((haze, None))
        else:
            matching_dict = dict()
            for haze in os.listdir(self.haze_dir):
                image = haze.split('_')[0] + '.' + haze.split('.')[-1]
                if image in matching_dict:
                    matching_dict[image].append(haze)
                else:
                    matching_dict[image] = [haze]
            for image in matching_dict:
                for haze in matching_dict[image]:
                    self.images.append((haze, image))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        haze_path = os.path.join(self.haze_dir, self.images[index][0])
        haze = Image.open(haze_path).convert("RGB")
        haze = haze if self.transform is None else self.transform(haze)

        image = []
        if self.image_dir is not None:
            image_path = os.path.join(self.image_dir, self.images[index][1])
            image = Image.open(image_path).convert("RGB")
            image = image if self.transform is None else self.transform(image)

        return haze, image
