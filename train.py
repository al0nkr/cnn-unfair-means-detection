import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.optim import Adam
from tqdm import tqdm

# Set parameters
num_classes = 6  # hand-raising, reading, etc.
batch_size = 32
num_epochs = 10
learning_rate = 1e-4

# Define image transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        width, height = image.size  # Get image dimensions

        # Load label
        label_path = os.path.join(self.labels_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')
        with open(label_path, 'r') as f:
            label_data = f.readline().strip().split()
            if not label_data:
                # Handle cases where label file is empty
                if self.transform:
                    image = self.transform(image)
                return image, num_classes  # Return an invalid class index
        
        # Parse label data
        class_label = int(label_data[0])
        x_center_norm = float(label_data[1])
        y_center_norm = float(label_data[2])
        width_norm = float(label_data[3])
        height_norm = float(label_data[4])
        
        # Convert normalized coordinates to absolute pixel coordinates
        x_center = x_center_norm * width
        y_center = y_center_norm * height
        bbox_width = width_norm * width
        bbox_height = height_norm * height
        
        # Calculate the coordinates of the bounding box
        x_min = int(x_center - bbox_width / 2)
        y_min = int(y_center - bbox_height / 2)
        x_max = int(x_center + bbox_width / 2)
        y_max = int(y_center + bbox_height / 2)
        
        # Ensure the coordinates are within image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        
        # Crop the image using the bounding box
        image_cropped = image.crop((x_min, y_min, x_max, y_max))
        
        if self.transform:
            image_cropped = self.transform(image_cropped)
        
        return image_cropped, class_label

if __name__ == '__main__':
    # Load datasets
    train_dataset = YoloDataset(images_dir=r'G:\cnn-detector\new_students_data\train\images', labels_dir=r'G:\cnn-detector\new_students_data\train\labels', transform=data_transforms)

    val_dataset = YoloDataset(images_dir=r'G:\cnn-detector\new_students_data\valid\images', labels_dir=r'G:\cnn-detector\new_students_data\valid\labels', transform=data_transforms)
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            num_workers=os.cpu_count())
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=os.cpu_count())

    # Load pretrained ResNet50 model and replace final layer
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            if (labels >= num_classes).any():
                continue  # Skip batches with invalid labels

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        correct, total = 0, 0
        val_loader_tqdm = tqdm(val_loader, desc="Validation")
        with torch.no_grad():
            for images, labels in val_loader_tqdm:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loader_tqdm.set_postfix(accuracy=100 * correct / total)
        
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

        # Save the model
        torch.save(model.state_dict(), f'action_detection_model{accuracy:.2f}%.pth')
