import asyncio
import io
import random

import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from torchvision import transforms

app = FastAPI()

# necesitas
# POST test (label: str, image: upload_file return true/false)
# GET label (return str)
# POST eval (image: upload_file return true false)
#
# Cada 5 minutos, un label distinto.
# Necesito un async

labels = [
    "acetic_acid",
    "benzene",
    "butane",
    "cyclohexane",
    "ethanol",
    "hexane",
    "formic_acid",
    "methanol",
    "octane",
    "propane",
]

current_label = labels[0]


@app.on_event("startup")
async def startup_event():
    print("Starting background label update task...")
    asyncio.create_task(update_label())


# 2. Define the SimpleCNN model (copied from train.py)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # Calculate input features for the first fully connected layer based on 224x224 input
        # 224 -> pool1 (112) -> pool2 (56)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# Constants for model loading and prediction
MODEL_SAVE_PATH = "image_classifier.pth"
# Ensure these match the order in train.py and your dataset folders
class_names = labels  # Using the same 'labels' list for class names

# 3. Load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = len(class_names)
model = SimpleCNN(num_classes=num_classes).to(device)

try:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded successfully from {MODEL_SAVE_PATH}")
except FileNotFoundError:
    print(
        f"Error: Model file not found at {MODEL_SAVE_PATH}. Please run train.py first."
    )
    # Exit or handle the error gracefully, maybe run a dummy model
except Exception as e:
    print(f"Error loading model: {e}")


# 4. Define preprocessing transforms for inference
def preprocess(pil_image) -> torch.Tensor:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(pil_image)


async def update_label():
    global current_label
    while True:
        await asyncio.sleep(1)
        current_label = random.choice(labels)


# No estoy seguro de como se estructura esta clase de archivos
# Se coloca primero variables, luego funciones, donde van las apis?

# Variables
# Funciones
# APIs


@app.post("/test")
async def test(label: str = Form(...), image: UploadFile = File(...)):
    if model is None:
        print("Model not loaded, cannot perform prediction.")
        return {"prediction": False, "message": "Model not loaded."}

    try:
        # Read the image bytes
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess the image
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0)

        # Move the input to the same device as the model
        input_batch = input_batch.to(device)

        with torch.no_grad():
            output = model(input_batch)

        # Get the predicted class
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        predicted_label = class_names[int(predicted_idx)]

        is_correct = predicted_label == label
        print(
            f"Received label: {label}, Predicted label: {predicted_label}, Correct: {is_correct}"
        )

        return {"prediction": is_correct, "predicted_label": predicted_label}

    except Exception as e:
        print(f"Error during prediction in /test endpoint: {e}")
        return {"prediction": False, "message": f"Error processing image: {e}"}


@app.get("/label")
async def get_label():
    return current_label


@app.post("/eval")
async def eval(image: UploadFile = File(...)):
    return True


if __name__ == "__main__":
    # Algun puerto que no interfiera
    uvicorn.run(app, host="0.0.0.0", port=3007)
