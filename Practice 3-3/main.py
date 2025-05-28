import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image 
import torch  
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms  
import torch.nn as nn 
import torch.optim as optim  
import torch.nn.functional as F
import matplotlib.pyplot as plt  

# تقسیم داده ها 
BASE_PATH = r"/home/snoou/project hosh /Practice 3-3"
DATA_PATH = os.path.join(BASE_PATH, "data")

image_path_list = []
label_list = []

for class_cat in os.listdir(DATA_PATH):
    class_path = os.path.join(DATA_PATH, class_cat)
    if os.path.isdir(class_path): 
        for image_object in os.listdir(class_path):
            if image_object.endswith(('.jpg', '.jpeg', '.png')): 
                image_path_list.append(os.path.join("data", class_cat, image_object))
                label_list.append(class_cat)

df = pd.DataFrame({"image_path": image_path_list, "label": label_list})

test_ratio = 0.20
train_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df['label'], random_state=42)

train_df.to_csv(os.path.join(BASE_PATH, "train.csv"), index=False)
test_df.to_csv(os.path.join(BASE_PATH, "test.csv"), index=False)

print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")
print(f"فایل‌های train.csv و test.csv در مسیر {BASE_PATH} ذخیره شدند.")

# dataset 


IMAGE_SIZE = 124 
train_transform = transforms.Compose([ 
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.RandomRotation(10), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])
test_transform = transforms.Compose([ 
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, class_list, transform=None, base_path=BASE_PATH):
        self.df = pd.read_csv(csv_file) 
        self.class_list = class_list  
        self.transform = transform
        self.base_path = base_path  

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.df.iloc[idx]['image_path']) 
        image = Image.open(img_path).convert('RGB')  
        label = self.class_list.index(self.df.iloc[idx]['label'])  
        if self.transform:
            image = self.transform(image) 
        return image, label 

CLASS_LIST = ['cloudy', 'desert', 'green_area', 'water']
train_dataset = SatelliteDataset(
    csv_file=os.path.join(BASE_PATH, "train.csv"),  
    class_list=CLASS_LIST,
    transform=train_transform  
)
test_dataset = SatelliteDataset(
    csv_file=os.path.join(BASE_PATH, "test.csv"), 
    class_list=CLASS_LIST,
    transform=test_transform  
)

BATCH_SIZE = 32  
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0  
)
test_loader = DataLoader(
    test_dataset,  
    batch_size=BATCH_SIZE,  
    shuffle=False, 
    num_workers=0 
)

# model 

class SatelliteImageClassifier(nn.Module):

    def __init__(self, num_classes, input_size=(124, 124), channels=3):
        super(SatelliteImageClassifier, self).__init__()
        self.input_size = input_size
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 

        self.bn1 = nn.BatchNorm2d(32)  
        self.bn2 = nn.BatchNorm2d(64)  
        self.bn3 = nn.BatchNorm2d(128)  
        self.bn4 = nn.BatchNorm2d(256)  

        self.pool = nn.MaxPool2d(2, 2) 

        self.dropout = nn.Dropout(0.5)

        self._to_linear = None
        self._calculate_to_linear(input_size)

        self.fc1 = nn.Linear(self._to_linear, 512)  
        self.fc2 = nn.Linear(512, num_classes) 

    def _calculate_to_linear(self, input_size):
        x = torch.randn(1, self.channels, *input_size)
        x = self.conv_forward(x)
        self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

    def conv_forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = self.pool(F.relu(self.bn4(self.conv4(x)))) 
        return x

    def forward(self, x):
        x = self.conv_forward(x)  
        x = x.view(-1, self._to_linear) 
        x = self.dropout(F.relu(self.fc1(x)))  
        x = self.fc2(x) 
        return x

NUM_CLASSES = len(CLASS_LIST)  
INPUT_SIZE = (124, 124)  
CHANNELS = 3  
LEARNING_RATE = 0.001  
EPOCHS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

model = SatelliteImageClassifier(NUM_CLASSES, INPUT_SIZE, CHANNELS).to(device) 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 


print("شروع آموزش مدل...")
for epoch in range(EPOCHS):
    model.train()  
    running_loss = 0.0  
    for images, labels in train_loader:  
        images, labels = images.to(device), labels.to(device) 
        optimizer.zero_grad()  
        outputs = model(images) 
        loss = criterion(outputs, labels)
        loss.backward()  
        optimizer.step() 
        running_loss += loss.item()  
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

    
torch.save(model.state_dict(), os.path.join(BASE_PATH, "satellite_model.pth"))
print("مدل ذخیره شد!")

# predict


def predict_image(image_path, model, transform, class_list):
    model.eval() 
    image = Image.open(image_path).convert('RGB') 
    image = transform(image).unsqueeze(0)  
    image = image.to(device)  
    with torch.no_grad(): 
        output = model(image) 
        _, predicted = torch.max(output, 1) 
        predicted_class = class_list[predicted.item()]  
    return predicted_class 


model = SatelliteImageClassifier(num_classes=len(CLASS_LIST), input_size=(124, 124), channels=3).to(device)
model.load_state_dict(torch.load(os.path.join(BASE_PATH, "satellite_model.pth")))  
model.eval()  

sample_image = os.path.join(DATA_PATH, "desert", os.listdir(os.path.join(DATA_PATH, "desert"))[0])  
predicted_class = predict_image(sample_image, model, test_transform, CLASS_LIST)  
image = Image.open(sample_image)  
plt.imshow(image) 
plt.title(f"دسته پیش‌بینی‌شده: {predicted_class}")  
plt.axis('off')  
plt.show()  
print(f"تصویر: {sample_image}")  
print(f"دسته پیش‌بینی‌شده: {predicted_class}") 