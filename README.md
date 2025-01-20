----------
# Sistema de Deteção de Resíduos Plásticos

Este projeto tem como objetivo a criação de um sistema inteligente para a deteção de resíduos plásticos em ambientes naturais, utilizando técnicas de Deep Learning. O sistema foi desenvolvido no âmbito da cadeira de Inteligência Artificial do curso de Engenharia Informática, sendo uma aplicação prática de modelos avançados para a preservação ambiental.

## Descrição do Projeto

O sistema utiliza um modelo pré-treinado da arquitetura **ResNet18** adaptado para a classificação binária entre "garrafa plástica" e "não garrafa plástica". A deteção é feita através de imagens, sendo as mesmas analisadas para determinar a presença de resíduos plásticos.

A aplicação possui uma interface gráfica em **Tkinter**, onde o utilizador pode carregar imagens, que são então processadas e analisadas pelo modelo de Deep Learning. O modelo foi treinado com um dataset de imagens, classificando as mesmas como "Garrafa Plástica" ou "Desconhecido".

## Tecnologias Utilizadas

-   **Python**: Linguagem de programação utilizada.
-   **PyTorch**: Biblioteca para Deep Learning, utilizada para treinar e realizar a inferência com o modelo.
-   **OpenCV**: Biblioteca para manipulação de imagens.
-   **Tkinter**: Framework para a criação da interface gráfica do utilizador.
-   **Torchvision**: Utilizada para trabalhar com redes neuronais pré-treinadas e realizar as transformações nas imagens.

## Estrutura do Repositório

```bash
├── dataset/
│   ├── train/
│   └── val/
├── treino.py               # Script para treinar o modelo
├── predictor.py            # Script para fazer a predição com o modelo treinado
├── interface.py            # Script para a interface gráfica
└── modelo_treinado.pth     # Modelo treinado e salvo

```

-   **dataset/**: Contém as pastas `train` e `val` com as imagens para treino e validação.
-   **treino.py**: Script responsável pelo treino do modelo.
-   **predictor.py**: Contém a função para carregar o modelo treinado e realizar a predição em novas imagens.
-   **interface.py**: Contém a implementação da interface gráfica para interação com o utilizador.
-   **modelo_treinado.pth**: Arquivo contendo o modelo treinado, pronto para realizar predições.

## Como Executar

### 1. Clonar o Repositório

Clone o repositório para a sua máquina local:

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git

```

### 2. Instalar Dependências

Instale as dependências necessárias utilizando o **pip**:

```bash
pip install -r requirements.txt

```

Aqui está um exemplo de `requirements.txt`:

```
torch==2.5.1
torchvision==0.20.1
opencv-python==4.11.0.86
Pillow==11.1.0
tk==0.1.0
```

### 3. Treinar o Modelo

Para treinar o modelo, execute o seguinte comando no terminal:

```bash
python treino.py

```

Isso irá treinar o modelo usando as imagens no diretório `dataset/` e salvar o modelo treinado como `modelo_treinado.pth`.

### 4. Executar a Interface Gráfica

Após o treino, pode-se executar a aplicação para testar a deteção de garrafas plásticas. Execute o seguinte comando:

```bash
python interface.py

```

A interface gráfica será aberta, permitindo que o utilizador selecione uma imagem. O sistema irá realizar a predição e mostrar o resultado.

## Resultados

Após a execução da interface, o sistema mostrará se uma garrafa plástica foi detectada ou não, com a probabilidade associada. A cor do texto indica se o sistema detectou ou não um plástico:

-   **Verde**: Garrafa Plástica detectada.
-   **Vermelho**: Nenhuma garrafa detectada.

## Contribuição

Se desejar contribuir para o projeto, siga os seguintes passos:

1.  Crie um fork do repositório.
2.  Crie uma nova branch (`git checkout -b feature/nova-funcionalidade`).
3.  Faça as alterações e commit (`git commit -am 'Adicionando nova funcionalidade'`).
4.  Faça push para a sua branch (`git push origin feature/nova-funcionalidade`).
5.  Crie uma pull request.

## Considerações Finais

Este projeto demonstra a aplicação de redes neuronais convolucionais no campo da preservação ambiental. Através da deteção de resíduos plásticos em imagens, o sistema pode ser usado como base para desenvolver soluções mais robustas para o monitoramento ambiental.

----------