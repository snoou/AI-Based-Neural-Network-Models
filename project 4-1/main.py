import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),  
    ])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),   
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)   
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2), 
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder().to(device) 
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 2
train_losses = []

print("Starting training...")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_idx, (images, _) in enumerate(train_loader): 
        
        images = images.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, images) 
        
        optimizer.zero_grad() 
        loss.backward()       
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0) 

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
    avg_epoch_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_epoch_loss)
    print(f'====> Epoch: {epoch+1} Average loss: {avg_epoch_loss:.4f}')

print("Training finished.")

plt.figure(figsize=(10,5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average MSE Loss')
plt.grid(True)
plt.savefig("training_loss_autoencoder.png") 
plt.show()

dataiter = iter(test_loader)
images, _ = next(dataiter) 
images_to_show = images[:10].to(device) 

model.eval() 
with torch.no_grad(): 
    reconstructed_images = model(images_to_show)

images_to_show = images_to_show.cpu()
reconstructed_images = reconstructed_images.cpu()

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for i in range(10):
    axes[0,i].imshow(np.transpose(images_to_show[i].squeeze().numpy(), (0,1)), cmap='gray')
    axes[0,i].get_xaxis().set_visible(False)
    axes[0,i].get_yaxis().set_visible(False)
    if i == 0:
        axes[0,i].set_title('Original Images', loc='left', fontsize=10)
        
    axes[1,i].imshow(np.transpose(reconstructed_images[i].squeeze().numpy(), (0,1)), cmap='gray')
    axes[1,i].get_xaxis().set_visible(False)
    axes[1,i].get_yaxis().set_visible(False)
    if i == 0:
        axes[1,i].set_title('Reconstructed Images', loc='left', fontsize=10)

plt.suptitle('Convolutional Autoencoder Results on MNIST', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97]) 
plt.savefig("reconstructed_images_autoencoder.png") 
plt.show()


