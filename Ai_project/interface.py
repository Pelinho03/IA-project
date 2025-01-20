import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from predictor import predict_image  # Importar a função atualizada do predictor.py

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

            # Converter a imagem de BGR (OpenCV) para RGB (para ser compatível com PIL e Tkinter)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Passar a imagem para a função de predição
            prob = predict_image(image_rgb)

            # Definir o resultado com base na probabilidade
            if prob > 0.5:  # Limite de confiança: 50%
                result_text = f"Detetado: Garrafa Plástica ({prob * 100:.2f}%)"
                result_color = "green"
            else:
                result_text = f"Não Detetado: ({prob * 100:.2f}%)"
                result_color = "red"

            # Atualizar o rótulo do resultado
            result_label.config(text=result_text, fg=result_color)

            # Redimensionar a imagem para caber na interface sem distorção, mantendo a proporção
            max_width = 500  # Largura máxima
            max_height = 400  # Altura máxima

            img_width, img_height = image_rgb.shape[1], image_rgb.shape[0]  # Dimensões originais da imagem
            scale_factor = min(max_width / img_width, max_height / img_height)  # Fator de escala

            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)

            resized_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Converter para formato compatível com Tkinter
            im_pil = Image.fromarray(resized_image)
            im_tk = ImageTk.PhotoImage(im_pil)

            # Atualizar a imagem na interface gráfica
            image_label.config(image=im_tk)
            image_label.image = im_tk  # Prevenir garbage collection da imagem

        except Exception as e:
            # Mostrar mensagem de erro detalhada
            messagebox.showerror("Erro", f"Erro ao processar a imagem: {str(e)}")
    else:
        messagebox.showinfo("Sem Arquivo", "Nenhuma imagem selecionada.")

# Criar a janela principal
root = tk.Tk()
root.title("Interface de Detecção de Garrafas Plásticas")
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

# Label para exibir a imagem submetida
image_label = tk.Label(root)
image_label.pack(pady=20)

# Iniciar a interface gráfica
root.mainloop()