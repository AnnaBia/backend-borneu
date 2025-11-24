# Borneu — Backend

Projeto backend desenvolvido como parte da Atividade Prática Supervisionada (APS) do curso de Bacharelado em Ciência da Computação. Este repositório contém a API Flask responsável pelo processamento e fornecimento de dados de previsão de desmatamento e NDVI para a ilha de Bornéu, consumidos pelo frontend disponível em outro repositório.  

**Repositório do frontend:** https://github.com/AnnaBia/frontend-borneu.git

## Tecnologias utilizadas

- **Python** 3.11 (recomenda-se usar `pyenv` ou `venv` para gerenciar versões)
- **Flask** 3.17
- **Flask-Cors** (para habilitar requisições cross-origin)
- **Pandas, NumPy, Scikit-learn, Joblib** (para processamento e modelagem de dados)
- **Openpyxl, CSV** (para leitura de séries históricas)

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/AnnaBia/backend-borneu.git
    ```
2. Navegue até o diretório do projeto:
    ```bash
    cd backend-borneu
    ```
3. Crie e ative um ambiente virtual (opcional, mas recomendado):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate
    ```
4. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

- Para iniciar a API em modo de desenvolvimento:
    ```bash
    python app.py
    ```
    O servidor Flask será iniciado na porta `http://localhost:5000`.

