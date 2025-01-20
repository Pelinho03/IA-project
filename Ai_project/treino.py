import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Transformações para pré-processamento das imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carregar os datasets
train_data = datasets.ImageFolder('dataset/train', transform=transform)
val_data = datasets.ImageFolder('dataset/val', transform=transform)

# DataLoaders para carregar os dados
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
val_loader = DataLoader(val_data, batch_size=5, shuffle=False)

# Modelo ResNet18 pré-treinado, ajustando para 3 classes (Humano, Garrafa, Background)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)  # Apenas 1 classe

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Critério e otimizador
criterion = nn.BCEWithLogitsLoss()  # Para problemas binários
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Função de treino
def train(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}')

        # Validar após cada época
        validate(model, val_loader)


# Função de validação
def validate(model, val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')


# Iniciar o treino
train(model, train_loader, val_loader, epochs=10)

# Salvar o modelo treinado
torch.save(model.state_dict(), 'modelo_treinado.pth')
print("Modelo salvo como 'modelo_treinado.pth'")
