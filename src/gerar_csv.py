import pandas as pd
import numpy as np

def gerar_csv(path="data/transacao.csv", n=5000):
    np.random.seed(42)
    df = pd.DataFrame({
        "amount": np.random.exponential(scale=200, size=n),
        "account_age_days": np.random.randint(30, 2000, size=n),
        "txn_hour": np.random.randint(0, 24, size=n),
        "channel": np.random.choice(["app", "web", "atm"], size=n),
        "country": np.random.choice(["BR", "US", "UK"], size=n),
        "is_fraud": np.random.choice([0, 1], size=n, p=[0.98, 0.02])
    })
    df.to_csv(path, index=False)
    print(f"Arquivo salvo em {path}")

if __name__ == "__main__":
    gerar_csv()