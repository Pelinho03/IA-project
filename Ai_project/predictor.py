import torch
import cv2
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np

# Configurar o dispositivo (GPU ou CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar o modelo treinado para classificação (ResNet18)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Ajustar para duas classes (Garrafa Plástica e Saca Plástica)
model.load_state_dict(torch.load('modelo_treinado.pth', map_location=device))  # Carregar pesos treinados
model = model.to(device)
model.eval()

# Carregar o modelo pré-treinado para detecção de objetos (Faster R-CNN)
detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
detection_model = detection_model.to(device)
detection_model.eval()

# Função de predição para imagem carregada
def predict_image(image, score_threshold=0.3):
    """
    Realiza a predição para determinar se a imagem contém um objeto e classifica-o.

    Args:
        image (numpy.ndarray): Imagem carregada no formato OpenCV.
        score_threshold (float): Limiar de confiança para as deteções.

    Returns:
        str: Classe com maior probabilidade (e.g., "Garrafa Plástica" ou "Saca Plástica").
        float: Confiança da classe predita.
        numpy.ndarray: Imagem com as caixas delimitadoras desenhadas.
    """
    # Converter a imagem de numpy array para PIL
    image_pil = Image.fromarray(image)

    # Transformar a imagem para o formato necessário para o Faster R-CNN
    transform = transforms.Compose([
        transforms.ToTensor()  # Converter para tensor
    ])
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # Detecção de objetos
    with torch.no_grad():
        prediction = detection_model(image_tensor)

    # Obter as caixas e a confiança (scores)
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Selecionar as caixas com uma confiança maior que o limiar fornecido
    selected_boxes = boxes[scores > score_threshold]

    # Converter imagem novamente para PIL para desenhar
    draw = ImageDraw.Draw(image_pil)

    # Desenhar as caixas na imagem
    for box in selected_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    # Transformação para preparar a imagem para a ResNet18 (classificação)
    transform_class = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionar para o tamanho compatível com ResNet18
        transforms.ToTensor(),  # Converter para tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor_class = transform_class(image_pil).unsqueeze(0).to(device)

    # Fazer predição com a ResNet18 (classificação)
    with torch.no_grad():
        output = model(image_tensor_class)
        probabilities = torch.softmax(output, dim=1)
        class_index = torch.argmax(probabilities).item()
        confidence = probabilities[0, class_index].item()

    # Mapear índice para o nome da classe
    threshold = 0.55  # Define um limite para considerar a predição confiável
    if probabilities[0, 0] > threshold:
        predicted_class = "Garrafa Plástica"
    elif probabilities[0, 1] > threshold:
        predicted_class = "Saca Plástica"
    else:
        predicted_class = "Indefinido"

    return predicted_class, confidence, np.array(image_pil), selected_boxes
