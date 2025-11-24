from src.gerar_csv import gerar_csv
from src.modelo import treinar_modelo
from src.predicao import prever
import pandas as pd

def main():
    # 1. Gerar dados
    print("Gerando arquivo de transações...")
    gerar_csv()

    # 2. Treinar modelo
    print("Treinando modelo de detecção de fraude...")
    treinar_modelo()

    # 3. Predição exemplo
    print("Fazendo previsão em 5 transações de exemplo:")
    df = pd.read_csv("data/transacao.csv").head(5).drop(columns=["is_fraud"])
    resultado = prever(df)
    print(resultado)

if __name__ == "__main__":
    main()