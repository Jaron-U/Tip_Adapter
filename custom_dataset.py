import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, shot_num=None):
        self.rood_dir = root_dir
        self.transform = transform
        self.shot_num = shot_num
        self.class_name = ["doughnut", "glass cup", "lemon", "chinese noodle", "chinese flute"]
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

if __name__ == "__main__":
    data_dir = "/home/jianglongyu/mydrive/clip_dataset/train_dataset"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def create_dataloaders(train_dir, test_dir, transform, shot_number=None, batch_size=32):
        train_dataset = CustomDataset(root_dir=train_dir, transform=transform, shot_num=shot_number)
        test_dataset = CustomDataset(root_dir=test_dir, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_loader, test_loader

    shot_number = 5
    batch_size = 25
    train_loader, test_loader = create_dataloaders(train_dir, test_dir, transform, shot_number, batch_size)

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels}")
        break