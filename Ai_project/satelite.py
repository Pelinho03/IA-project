import torch
import cv2
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Carregar o modelo treinado
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Função de predição para imagem carregada
def predict_image(image):
    # Converter a imagem de numpy array para PIL
    image_pil = Image.fromarray(image)

    # Definir transformações para a imagem (necessário para o modelo)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Aplicar transformações
    image_tensor = transform(image_pil).unsqueeze(0)  # Adicionar uma dimensão para o batch

    # Previsão com o modelo
    with torch.no_grad():
        prediction = model(image_tensor)

    # Obter caixas, rótulos e pontuações
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    return boxes, labels, scores
