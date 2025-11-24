import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Constants
# Prob doesn't need change
DATA_DIR = "dataset/"
MODEL_SAVE_PATH = "image_classifier.pth"
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 1. Define data transforms
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# 2. Load the dataset
print("Loading dataset...")
image_dataset = datasets.ImageFolder(DATA_DIR, data_transforms["train"])
dataloader = DataLoader(
    image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
dataset_size = len(image_dataset)
class_names = image_dataset.classes
num_classes = len(class_names)
print(f"Found {dataset_size} images belonging to {num_classes} classes: {class_names}")


# 3. Define a simple CNN model
# Conv2d is convolutional layer for images. Very important.
# ReLU is activation functions. Converts the model from a linear function into an actual model
# MaxPool2d Pooling Layers. Basically reduce the images to it's most significant pixels
# Flatten. Converts the mess of dimmensions into a simple long 1d vector, useful for:
# Linear. Regular Neural Network layers. The selector.
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Insert image here
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Better than sigmoid
        self.relu1 = nn.ReLU()
        # Get the most significant pixels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Basically increase the number of layers there are
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Apanado
        self.flatten = nn.Flatten()
        # Calculate input features for the first fully connected layer based on 224x224 input
        # 224 -> pool1 (112) -> pool2 (56)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        # The number of possible outputs is always the number of classes
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 4. Set up training
# Determine if using CUDA (gpu) or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Finally create the model
model = SimpleCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
# Adam. Always adam for some reason.
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. Training loop
# The usual.
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / dataset_size
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} Loss: {epoch_loss:.4f}")

# 6. Save the trained model
print("Training complete. Saving model...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
