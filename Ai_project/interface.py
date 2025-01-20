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
            image = cv2.imread(file_path)

            if image is None:
                raise ValueError("Erro ao carregar a imagem. Verifica se o arquivo está no formato suportado.")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            prob = predict_image(image_rgb)

            if prob > 0.5:
                result_text = f"Detetado: Garrafa Plástica ({prob * 100:.2f}%)"
                result_color = "green"
            else:
                result_text = f"Não Detetado: ({prob * 100:.2f}%)"
                result_color = "red"

            result_label.config(text=result_text, fg=result_color)

            max_width = 500
            max_height = 400

            img_width, img_height = image_rgb.shape[1], image_rgb.shape[0]
            scale_factor = min(max_width / img_width, max_height / img_height)

            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)

            resized_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

            im_pil = Image.fromarray(resized_image)
            im_tk = ImageTk.PhotoImage(im_pil)

            image_label.config(image=im_tk)
            image_label.image = im_tk

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar a imagem: {str(e)}")
    else:
        messagebox.showinfo("Sem Arquivo", "Nenhuma imagem selecionada.")

# Criar a janela principal
root = tk.Tk()
root.title("Deteção de Garrafas Plásticas")
root.geometry("800x600")
root.config(bg="#f2f2f2")  # Cor de fundo suave

# Frame para agrupar os componentes principais
frame = tk.Frame(root, bg="#f2f2f2")
frame.pack(fill="both", expand=True)

# Título
title_label = tk.Label(frame, text="Sistema de Deteção de Garrafas Plásticas", font=("Ubuntu", 24, "bold"), bg="#f2f2f2")
title_label.pack(pady=20)

# Texto de instrução
instruction_label = tk.Label(frame, text="Escolhe uma imagem para análise", font=("Ubuntu", 14), bg="#f2f2f2", fg="#555")
instruction_label.pack(pady=10)

# Botão para selecionar o arquivo
select_button = tk.Button(frame, text="Selecionar Imagem", font=("Ubuntu", 12), bg="#4CAF50", fg="white", relief="flat", padx=20, pady=10, command=open_file)
select_button.pack(pady=20)

# Label para mostrar o resultado da predição
result_label = tk.Label(frame, text="", font=("Ubuntu", 14), bg="#f2f2f2")
result_label.pack(pady=10)

# Label para exibir a imagem submetida
image_label = tk.Label(frame, bg="#f2f2f2")
image_label.pack(pady=20)

# Iniciar a interface gráfica
root.mainloop()
