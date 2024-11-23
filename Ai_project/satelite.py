import torch
from torchvision import transforms, models
import cv2
import numpy as np

# Configurar dispositivo (GPU ou CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar o modelo Faster R-CNN pré-treinado
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)

# Transformação da imagem para o formato aceito pelo modelo
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),  # Ajuste o tamanho conforme necessário
    transforms.ToTensor(),
])

# Função para realizar predições
def predict_image(image):
    # Aplicar transformações e enviar a imagem para o dispositivo
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Fazer a predição com o modelo
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extrair as caixas, labels e scores das detecções
    scores = outputs[0]['scores'].cpu().numpy()
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    return boxes, labels, scores

# Carregar a imagem a partir de um arquivo
image_path = 'caminho/para/sua/imagem.jpg'  # Substitua pelo caminho correto da imagem
image = cv2.imread(image_path)

# Verificar se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
    exit()

# Realizar predição na imagem carregada
boxes, labels, scores = predict_image(image)

# Processar as detecções
for box, label, score in zip(boxes, labels, scores):
    if score > 0.3:  # Filtrar detecções com confiança maior que 0.3
        x1, y1, x2, y2 = map(int, box)  # Converter coordenadas para inteiros

        # Aqui vamos mostrar os valores de score e label para depuração
        print(f"Classe: {label}, Score: {score:.2f}")  # Imprimir valores de debug
        print(f"Caixa: {box}")

        # Mapeamento das classes: verifique as classes para o seu caso
        if label == 1:  # 'Humano'
            class_name = 'Humano'
        elif label == 43:  # 'Garrafa'
            class_name = 'Garrafa'
        else:
            class_name = 'Desconhecido'  # Para outras classes não esperadas

        # Desenhar a caixa delimitadora
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Inserir o texto com a classe e confiança
        cv2.putText(image, f'{class_name}: {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Exibir a imagem com as detecções
cv2.imshow("Detecção na Imagem", image)

# Salvar a imagem processada com as caixas de detecção
output_path = 'imagem_detectada.jpg'
cv2.imwrite(output_path, image)
print(f"Imagem salva como {output_path}")

# Aguardar o usuário fechar a janela de visualização
cv2.waitKey(0)
cv2.destroyAllWindows()
