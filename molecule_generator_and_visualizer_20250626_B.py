import torch
import torch.nn as nn
import torch.optim as optim
import random
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Force CPU use
device = torch.device("cpu")

# Generator model definition
class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1024)  # Output matches fingerprint size

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Discriminator model definition
class DiscriminatorModel(nn.Module):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# === Load SMILES and fingerprint data ===
fingerprints = np.load("C:/Users/Kelly/Dropbox/03_Summer_2025/ML_Workshop/kaggle_smiles/kaggle_fps.npy")
smiles_list = np.load("C:/Users/Kelly/Dropbox/03_Summer_2025/ML_Workshop/kaggle_smiles/kaggle_smiles.npy", allow_pickle=True).tolist()

# === SMILES generator ===
def generate_smiles(model, top_k=10):
    z = torch.randn(1, 100).to(device)
    with torch.no_grad():
        generated_fp = model(z).cpu().numpy()

    similarities = cosine_similarity(generated_fp, fingerprints)
    top_indices = np.argsort(similarities[0])[-top_k:]
    idx = random.choice(top_indices)
    return smiles_list[idx]

# Check if the SMILES string is valid
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Visualization function
def visualize_smiles(smiles_list):
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list if is_valid_smiles(smi)]
    img = Draw.MolsToGridImage(molecules, molsPerRow=3, subImgSize=(200, 200))
    img.show()

# === Training function ===
def train_gan(generator, discriminator, num_epochs=20, batch_size=32, lr=0.0002):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    loss_d_values = []
    loss_g_values = []

    for epoch in range(num_epochs):
        for _ in range(batch_size):
            real_labels = torch.ones(1, 1).to(device)
            fake_labels = torch.zeros(1, 1).to(device)

            idx = np.random.randint(0, len(fingerprints))
            real_data = torch.tensor(fingerprints[idx], dtype=torch.float32).unsqueeze(0).to(device)

            z = torch.randn(1, 100).to(device)
            fake_data = generator(z)

            # Train discriminator
            output_real = discriminator(real_data)
            loss_real = criterion(output_real, real_labels)

            output_fake = discriminator(fake_data.detach())
            loss_fake = criterion(output_fake, fake_labels)

            loss_d = (loss_real + loss_fake) / 2
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # Train generator
            output_fake = discriminator(fake_data)
            loss_g = criterion(output_fake, real_labels)

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}')
        loss_d_values.append(loss_d.item())
        loss_g_values.append(loss_g.item())
    return loss_d_values, loss_g_values
# === Main block ===
if __name__ == "__main__":
    generator = GeneratorModel().to(device)
    discriminator = DiscriminatorModel().to(device)

    #train_gan(generator, discriminator, num_epochs=100, batch_size=16)
    loss_d_values, loss_g_values = train_gan(generator, discriminator, num_epochs=100, batch_size=16)

# plot training data
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(loss_d_values) + 1))
    plt.plot(epochs, loss_d_values, label="Discriminator Loss")
    plt.plot(epochs, loss_g_values, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gan_loss_plot.png", dpi=300)
    plt.show()



    # Generate SMILES strings
    num_samples = 50
    generated_smiles = [generate_smiles(generator, top_k=1000) for _ in range(num_samples)]

    print("All Generated SMILES:")
    for smi in generated_smiles:
        print(smi)

    valid_smiles = [smi for smi in generated_smiles if is_valid_smiles(smi)]
    invalid_smiles = [smi for smi in generated_smiles if not is_valid_smiles(smi)]

    print(f"Valid SMILES ({len(valid_smiles)}): {valid_smiles}")
    if invalid_smiles:
        print(f"Invalid SMILES ({len(invalid_smiles)}): {invalid_smiles}")

    if valid_smiles:
        visualize_smiles(valid_smiles)
