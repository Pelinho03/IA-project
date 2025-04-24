
# **OceanEye**  

Sistema de Deteção de Resíduos Plásticos com Inteligência Artificial  

Este repositório contém a implementação de uma aplicação desenvolvida no âmbito da unidade curricular de **Inteligência Artificial**, do curso de **Engenharia Informática**. O projeto tem como objetivo contribuir para a preservação ambiental através da deteção automática de garrafas e sacas plásticas em imagens, utilizando técnicas avançadas de Deep Learning.

----------

## **Objetivo**  

O principal objetivo deste projeto é:  

- Desenvolver um sistema inteligente para **deteção e classificação de resíduos plásticos** em imagens.  
- Implementar um modelo de **Deep Learning** baseado em redes neuronais convolucionais (CNN).  
- Criar uma **interface gráfica intuitiva** para interação com o utilizador.  
- Aplicar conceitos de **visão computacional**, **aprendizagem profunda** e **processamento de imagens** em um cenário real.  

----------

## **Funcionalidades**  

### **Deteção de Objetos**  
- Identifica garrafas e sacas plásticas em imagens utilizando **Faster R-CNN**.  
- Desenha **caixas delimitadoras** nos objetos detectados.  

### **Classificação de Resíduos**  
- Classifica os objetos detectados em duas categorias:  
  - **Garrafa Plástica**  
  - **Saca Plástica**  
- Exibe a **confiança da predição** em percentagem.  

### **Interface Gráfica**  
- Permite ao utilizador **carregar imagens** para análise.  
- Mostra o resultado da deteção e classificação em tempo real.  
- Interface desenvolvida com **Tkinter**, com design moderno e responsivo.  

----------

## **Tecnologias Utilizadas**  

- **Python 3** – Linguagem principal do projeto.  
- **PyTorch** – Framework para Deep Learning (treino e inferência).  
- **OpenCV** – Processamento de imagens e deteção de objetos.  
- **Torchvision** – Acesso a modelos pré-treinados e transformações de imagem.  
- **Tkinter** – Interface gráfica do utilizador (GUI).  
- **Pillow (PIL)** – Manipulação e exibição de imagens.  

----------

## **Estrutura do Projeto**  

```bash
OceanEye/
│
├── dataset/                # Pasta com dados de treino e validação
│   ├── train/              # Imagens para treino
│   └── val/                # Imagens para validação
│
├── treino.py               # Script para treinar o modelo
├── predictor.py            # Lógica de predição (Faster R-CNN + ResNet18)
├── interface.py            # Interface gráfica (Tkinter)
│
├── modelo_treinado.pth     # Modelo treinado (ResNet18)
└── README.md               # Documentação do projeto
```

----------

## **Como Executar o Projeto**  

### **1. Clonar o Repositório**  
```bash
git clone https://github.com/Pelinho03/IA-project.git
cd IA-project
```

### **2. Instalar Dependências**  
```bash
pip install -r requirements.txt
```

**Exemplo de `requirements.txt`:**  
```
torch==2.5.1
torchvision==0.20.1
opencv-python==4.11.0.86
Pillow==11.1.0
tk==0.1.0
```

### **3. Treinar o Modelo (Opcional)**  
Se desejar treinar o modelo novamente:  
```bash
python treino.py
```

### **4. Executar a Aplicação**  
```bash
python interface.py
```

----------

## **Resultados e Exemplos**  

- **Deteção bem-sucedida:**  
  - A interface exibe a imagem com as **caixas delimitadoras** e a classe detectada.  
  - **Cores de feedback:**  
    - ✅ **Verde**: Garrafa Plástica detectada.  
    - 🔵 **Azul**: Saca Plástica detectada.  
    - ❌ **Vermelho**: Nenhum plástico detectado.  

----------

## **Ideia Central**  

> Este projeto demonstra como a **Inteligência Artificial** pode ser aplicada em desafios ambientais, fornecendo uma ferramenta para deteção automática de poluição por plásticos. A solução pode ser integrada em sistemas de monitorização costeira ou robótica de limpeza.  

----------

## **Desenvolvido por**  

**Paulo Guimarães**  
[GitHub](https://github.com/Pelinho03)  
