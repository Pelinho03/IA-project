import torch
import cv2
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Configurar o dispositivo (GPU ou CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar o modelo treinado
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Ajustar para uma única classe
model.load_state_dict(torch.load('modelo_treinado.pth', map_location=device))  # Carregar pesos treinados
model = model.to(device)
model.eval()

# Função de predição para imagem carregada
def predict_image(image):
    """
    Realiza a predição para determinar se a imagem contém uma garrafa plástica.

    Args:
        image (numpy.ndarray): Imagem carregada no formato OpenCV.

    Returns:
        float: Probabilidade de a imagem conter uma garrafa plástica.
    """
    # Converter a imagem de numpy array para PIL
    image_pil = Image.fromarray(image)

    # Transformar a imagem
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionar para o tamanho compatível com ResNet18
        transforms.ToTensor(),  # Converter para tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalização padrão
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # Adicionar dimensão de batch e enviar para o dispositivo

    # Fazer predição
    with torch.no_grad():
        output = model(image_tensor)  # Passar pela rede
        prob = torch.sigmoid(output).item()  # Converter logits para probabilidade usando sigmoid

    return prob
