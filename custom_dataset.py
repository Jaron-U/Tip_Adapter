import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

classnames = ["doughnut", "glass cup", "lemon", "chinese noodle", "chinese flute"]
template = ["a photo of {}"]
label2name = {
            0: "doughnut",
            1: "glass_cup",
            2: "lemon",
            3: "chinese_noodle",
            4: "chinese_flute",
            5: "others"
        }

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, shot_num=None):
        self.rood_dir = root_dir
        self.transform = transform
        self.shot_num = shot_num
        self.classnames = classnames
        self.template = template
        self.label2name = classnames
        self.image_paths = []
        self.labels = []

        for class_dir in os.listdir(root_dir):
            label = int(class_dir.split("_")[0])
            class_dir = os.path.join(root_dir, class_dir)
            image_files = os.listdir(class_dir)

            if shot_num:
                image_files = image_files[:shot_num]
            
            for img_file in image_files:
                self.image_paths.append(os.path.join(class_dir, img_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image, label

def getDataset(data_dir, preprocess, shot_num=None):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), 
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                             std=(0.26862954, 0.26130258, 0.27577711))
        ])

    train_dataset = CustomDataset(root_dir=train_dir, transform=transform, shot_num=shot_num)
    test_dataset = CustomDataset(root_dir=test_dir, transform=preprocess)

    return train_dataset, test_dataset