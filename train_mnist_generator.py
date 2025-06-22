import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DigitGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, noise_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = noise * self.label_embed(labels)
        return self.model(x).view(-1, 1, 28, 28)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitGenerator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    loss_fn = nn.MSELoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1)  # normalize to [-1, 1]
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(5):  # keep training short for quick testing
        for imgs, labels in loader:
            noise = torch.randn(imgs.size(0), 100).to(device)
            labels = labels.to(device)
            fake_imgs = model(noise, labels)
            loss = loss_fn(fake_imgs, imgs.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "digit_generator.pth")

if __name__ == "__main__":
    train_model()
