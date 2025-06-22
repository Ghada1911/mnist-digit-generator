import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

@st.cache_resource
def load_model():
    model = DigitGenerator()
    model.load_state_dict(torch.load("digit_generator.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

def generate_images(model, digit):
    noise = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        generated = model(noise, labels)
    return generated

st.title("MNIST Digit Generator")
digit = st.selectbox("Choose a digit (0-9):", list(range(10)))

if st.button("Generate"):
    model = load_model()
    images = generate_images(model, digit)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i].squeeze().numpy(), cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
