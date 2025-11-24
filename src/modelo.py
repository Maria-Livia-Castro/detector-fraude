import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.processamento import carregar_dados, criar_preprocessador

def treinar_modelo():
    df = carregar_dados()
    target = "is_fraud"
    cat_features = [c for c in df.columns if df[c].dtype == "object"]
    num_features = [c for c in df.columns if c not in cat_features + [target]]

    X = df[num_features + cat_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    preprocess = criar_preprocessador(num_features, cat_features)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipe = Pipeline([("prep", preprocess), ("clf", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.show()

    joblib.dump(pipe, "fraud_model.joblib")
    print("Modelo salvo em fraud_model.joblib")

if __name__ == "__main__":
    treinar_modelo()