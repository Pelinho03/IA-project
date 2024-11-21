import torch
from torchvision import transforms, models
import cv2
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])


classes = ['__background__', 'plastico', 'nuvens', 'terra', 'mar']

def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    scores = outputs[0]['scores'].cpu().numpy()
    boxes = outputs[0]['boxes'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()

    return boxes, labels, scores


cap = cv2.VideoCapture(0)

print("Pressione 'q' para sair da detecção em tempo real.")
while True:
    ret, frame = cap.read()
    if not ret:
        break


    boxes, labels, scores = predict_image(frame)


    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box)

            if label < len(classes):
                class_name = classes[label]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f'{class_name}: {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    cv2.imshow("CAMERA SATELITE", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
