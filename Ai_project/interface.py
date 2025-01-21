import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from predictor import predict_image  # Importar a função atualizada do predictor.py

# Função para abrir o diálogo e selecionar uma imagem
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Arquivos de Imagem", "*.jpg;*.jpeg;*.png")])

    if file_path:
        try:
            # Carregar a imagem usando OpenCV
            image = cv2.imread(file_path)

            if image is None:
                raise ValueError("Erro ao carregar a imagem. Verifica se o arquivo está no formato suportado.")

            # Converter para RGB (OpenCV usa BGR por padrão)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Fazer a predição usando o modelo
            class_label, confidence, output_image, boxes = predict_image(image_rgb)

            # Resultado textual com a confiança
            result_text = f"{class_label} ({confidence * 100:.2f}%)"
            result_color = "green" if class_label == "Garrafa Plástica" else "blue"

            result_label.config(text=result_text, fg=result_color)

            # Redimensionar a imagem para caber na interface sem perder proporção
            max_width = 500
            max_height = 400

            img_width, img_height = output_image.shape[1], output_image.shape[0]
            scale_factor = min(max_width / img_width, max_height / img_height)

            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)

            resized_image = cv2.resize(output_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Converter para o formato que o Tkinter suporta
            im_pil = Image.fromarray(resized_image)
            im_tk = ImageTk.PhotoImage(im_pil)

            # Atualizar a imagem exibida na interface
            image_label.config(image=im_tk)
            image_label.image = im_tk

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar a imagem: {str(e)}")
    else:
        messagebox.showinfo("Sem Arquivo", "Nenhuma imagem selecionada.")


# Criar a janela principal
root = tk.Tk()
root.title("Deteção de Garrafas e Sacas plásticas")
root.geometry("900x700")
root.config(bg="#f2f2f2")  # Cor de fundo suave

# Frame para agrupar os componentes principais
frame = tk.Frame(root, bg="#F8F8F2")
frame.pack(fill="both", expand=True)

# Título
title_label = tk.Label(frame, text="Sistema de Deteção de Garrafas e Sacas plásticas", font=("Ubuntu", 20, "bold"), bg="#F8F8F2", fg="#282A36")
title_label.pack(pady=20)

# Texto de instrução
instruction_label = tk.Label(frame, text="Escolhe uma imagem para análise", font=("Ubuntu", 14), bg="#F8F8F2", fg="#282A36")
instruction_label.pack(pady=10)

# Botão para selecionar o arquivo
select_button = tk.Button(frame, text="Selecionar Imagem", font=("Ubuntu", 12), bg="#282A36", fg="#F8F8F2", relief="flat", padx=20, pady=10, command=open_file)
select_button.pack(pady=20)

# Label para mostrar o resultado da predição
result_label = tk.Label(frame, text="", font=("Ubuntu", 16), bg="#F8F8F2", fg="#282A36")
result_label.pack(pady=10)

# Label para exibir a imagem submetida
image_label = tk.Label(frame, bg="#f2f2f2")
image_label.pack(pady=20)

# Rodapé com instruções adicionais
footer_label = tk.Label(root, text="Desenvolvido por Daniel | Paulo | Rúben", font=("Ubuntu", 10), bg="#F8F8F2", fg="#282A36")
footer_label.pack(side="bottom", pady=10)

# Iniciar a interface gráfica
root.mainloop()
