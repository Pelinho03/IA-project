import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from satelite import predict_image  # Função de predição do satélite


# Função para abrir o diálogo e selecionar uma imagem
def open_file():
    # Abrir um diálogo para selecionar o arquivo de imagem
    file_path = filedialog.askopenfilename(filetypes=[("Arquivos de Imagem", "*.jpg;*.jpeg;*.png")])

    if file_path:
        try:
            # Carregar a imagem com o OpenCV
            image = cv2.imread(file_path)

            if image is None:
                raise ValueError("Erro ao carregar a imagem. Verifica se o arquivo está no formato suportado.")

            # Converter a imagem de BGR (OpenCV) para RGB (para ser compatível com PIL e o Tkinter)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Passar a imagem para a função de predição
            boxes, labels, scores = predict_image(image_rgb)  # Passando diretamente a imagem do OpenCV

            # Inicializar a variável para armazenar a classe mais provável
            class_name = "Desconhecido"
            found_detection = False

            # Imprimir as detecções para diagnóstico
            print(f"Caixas de detecção: {boxes}")
            print(f"Rótulos: {labels}")
            print(f"Pontuações: {scores}")

            # Iterar pelas deteções e verificar a confiança
            for label, score, box in zip(labels, scores, boxes):
                if score > 0.4:  # Considerar apenas deteções com confiança > 0.2
                    found_detection = True
                    # Associações de labels
                    if label == 1:
                        class_name = 'Humano'
                        color = (0, 255, 0)  # Cor verde para Humanos
                    elif label == 43:
                        class_name = 'Mar'
                        color = (255, 0, 0)  # Cor azul para Mar
                    elif label == 44:
                        class_name = 'Garrafa'
                        color = (0, 0, 255)  # Cor vermelha para Garrafa
                    elif label == 45:
                        class_name = 'Nuvens'
                        color = (255, 255, 0)  # Cor amarela para Nuvens
                    elif label == 46:
                        class_name = 'Plástico'
                        color = (255, 0, 255)  # Cor rosa para Plástico
                    elif label == 47:
                        class_name = 'Terra'
                        color = (0, 255, 255)  # Cor ciano para Terra
                    else:
                        class_name = 'Desconhecido'  # Caso o label não corresponda a nenhuma classe conhecida
                        color = (255, 255, 255)  # Cor branca para Detecção Desconhecida

                    # Desenhar a caixa de detecção na imagem com a cor correspondente
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Caso não tenha encontrado nenhuma detecção com confiança
            if not found_detection:
                class_name = "Nenhuma detecção confiável"

            # Exibir o nome da classe detectada
            result_label.config(text=f'Resultado: {class_name}')

            # Redimensionar a imagem para caber na janela sem distorção, mantendo a proporção
            max_width = 500  # Ajuste conforme necessário
            max_height = 400  # Ajuste conforme necessário

            # Obter as dimensões originais da imagem
            img_width, img_height = image_rgb.shape[1], image_rgb.shape[0]

            # Calcular o fator de escala para manter a proporção
            scale_factor = min(max_width / img_width, max_height / img_height)

            # Calcular as novas dimensões da imagem
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)

            # Redimensionar a imagem sem distorcer, mantendo a proporção
            im_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Converter para ImageTk (no formato PIL)
            im_pil = Image.fromarray(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
            im_tk = ImageTk.PhotoImage(im_pil)

            # Atualizar a imagem na interface gráfica
            image_label.config(image=im_tk)
            image_label.image = im_tk  # Para evitar que a imagem seja descartada

        except Exception as e:
            # Mostrar mensagem de erro detalhada
            messagebox.showerror("Erro", f"Erro ao processar a imagem: {str(e)}")
    else:
        messagebox.showinfo("Sem Arquivo", "Nenhuma imagem selecionada.")


# Criar a janela principal
root = tk.Tk()
root.title("Interface de Detecção de Imagens")
root.geometry("800x600")

# Texto de instrução
instruction_label = tk.Label(root, text="Escolhe uma imagem", font=("Ubuntu", 14))
instruction_label.pack(pady=20)

# Botão para selecionar o arquivo
select_button = tk.Button(root, text="Selecionar Imagem", font=("Ubuntu", 12), command=open_file)
select_button.pack(pady=10)

# Label para mostrar o resultado da predição
result_label = tk.Label(root, text="", font=("Ubuntu", 12))
result_label.pack(pady=10)

# Label para exibir a imagem com as caixas de detecção
image_label = tk.Label(root)
image_label.pack(pady=20)

# Iniciar a interface gráfica
root.mainloop()
