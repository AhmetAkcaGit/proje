#%% Kullanılacak kütüphaneler
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, matthews_corrcoef
import seaborn as sns
import time
import os

#%% Cihaz
cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(cihaz)

#%% Veri önişleme
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#%% Veri yükleme ve bölme
dataset = datasets.ImageFolder(root=r"C:\Users\ASUS ROG\.spyder-py3\udemy\Derin Öğrenme\kanserArchive\train\zoomzus", transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
print(len(train_loader), len(train_dataset))
print(len(test_loader), len(test_dataset))

#%% Hiperparametreler
latent_dim = 100
lr = 0.0001
beta1 = 0.5
beta2 = 0.999
num_epochs = 500
image_save_interval = 5  # Her 5 epokda bir görüntü kaydet

#%% Üretici
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img
    
#%% Ayrımcı
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(512*2*2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity
    
#%% Üretici ve ayrımcı tanımlama ve başlatma
generator = Generator(latent_dim).to(cihaz)
discriminator = Discriminator().to(cihaz)
adverserial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Görüntüleri kaydedeceğimiz dizin
os.makedirs("generated_images", exist_ok=True)

#%% Eğitim döngüsü
baslangic = time.time()
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        real_images = batch[0].to(cihaz)
        valid = torch.ones(real_images.size(0), 1, device=cihaz)
        fake = torch.zeros(real_images.size(0), 1, device=cihaz)
        real_images = real_images.to(cihaz)
        
        # Ayrımcı eğitimi
        optimizer_D.zero_grad()
        z = torch.randn(real_images.size(0), latent_dim, device=cihaz)
        fake_images = generator(z)
        real_loss = adverserial_loss(discriminator(real_images), valid)
        fake_loss = adverserial_loss(discriminator(fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        # Üretici eğitimi
        optimizer_G.zero_grad()
        gan_images = generator(z)
        g_loss = adverserial_loss(discriminator(gan_images), valid)
        g_loss.backward()
        optimizer_G.step()
        
        # Görüntüleme
        if (i + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(train_loader)} Ayrımcı Kaybı: {d_loss.item():.4f} Üretici Kaybı: {g_loss.item():.4f}")
    
    # Her 5 epokda bir görüntü kaydı
    if (epoch + 1) % image_save_interval == 0:
        with torch.no_grad():
            generator.eval()
            z = torch.randn(1, latent_dim, device=cihaz)
            sample_images = generator(z)
            sample_images = sample_images * 0.5 + 0.5
            grid = torchvision.utils.make_grid(sample_images, nrow=8)
            torchvision.utils.save_image(grid, f"generated_images/epoch_{epoch+1}.png")
            generator.train()

bitis = time.time()
print("İşlem süresi ->", str(bitis - baslangic))

# Test seti üzerinde değerlendirme
all_real_labels = []
all_predicted_labels = []
with torch.no_grad():
    discriminator.eval()
    for batch in test_loader:
        real_images = batch[0].to(cihaz)
        valid = torch.ones(real_images.size(0), 1, device=cihaz)
        fake = torch.zeros(real_images.size(0), 1, device=cihaz)
        
        z = torch.randn(real_images.size(0), latent_dim, device=cihaz)
        fake_images = generator(z)
        
        real_output = discriminator(real_images).view(-1)
        fake_output = discriminator(fake_images).view(-1)
        
        predicted_labels = torch.cat((real_output, fake_output)).cpu().numpy()
        true_labels = torch.cat((valid.cpu(), fake.cpu())).numpy()
        
        all_real_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_labels)

# Performans metriklerini hesaplama
accuracy = accuracy_score(all_real_labels, np.round(all_predicted_labels))
f1 = f1_score(all_real_labels, np.round(all_predicted_labels))
precision = precision_score(all_real_labels, np.round(all_predicted_labels))
recall = recall_score(all_real_labels, np.round(all_predicted_labels))
mcc = matthews_corrcoef(all_real_labels, np.round(all_predicted_labels))
conf_matrix = confusion_matrix(all_real_labels, np.round(all_predicted_labels))

# Sonuçlar
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("MCC:", mcc)
print("Confusion Matrix:")
print(conf_matrix)

# Confusion matrix görselleştirme
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.title('Confusion Matrix')
plt.show()