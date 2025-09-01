import argparse
import csv

import numpy as np

#Parseo de linea de comando

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regresión lineal")
    p.add_argument("--data", required=True, help="Ruta al CSV")
    p.add_argument("--target-index", type=int, default=-1, help="Índice de la columna objetivo (default: última).")
    p.add_argument("--header", action="store_true", help="Indica si el CSV tiene encabezado.")
    p.add_argument("--test-size", type=float, default=0.2, help="Proporción del conjunto de prueba (0-1).")
    p.add_argument("--seed", type=int, default=42, help="Semilla para el split.")
    p.add_argument("--method", choices=["normal", "gd"], default="normal", help="Método de entrenamiento: ecuación normal o gradiente.")
    p.add_argument("--standardize", action="store_true", help="Estandarizar X con media/desv del train.")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate (GD).")
    p.add_argument("--epochs", type=int, default=2000, help="Épocas (GD).")
    p.add_argument("--predict", type=str, default=None, help='Vector para predecir (sin target).')
    return p.parse_args()


# Carga y partición de datos

def read_csv(path: str, header: bool, target_idx: int):
    # TODO: Implementar lectura real del CSV
    raise NotImplementedError("TODO")


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float, seed: int):
    # TODO: Implementar partición aleatoria 
    raise NotImplementedError("TODO")


# Preprocesamiento

def standardize_fit(X: np.ndarray):
    # TODO: Implementar cálculo de medias y desviaciones
    raise NotImplementedError("TODO")


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # TODO: Implementar aplicación de estandarización
    raise NotImplementedError("TODO")


def add_bias(X: np.ndarray) -> np.ndarray:
    # TODO: Implementar concatenación de columna de 1s
    raise NotImplementedError("TODO")


# Métricas

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO: Implementar MSE
    raise NotImplementedError("TODO")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO: Implementar R^2
    raise NotImplementedError("TODO")


# Modelo

class LinearRegressor:
    def __init__(self, method="normal", lr=0.01, epochs=2000, standardize=False):
        self.method = method
        self.lr = lr
        self.epochs = epochs
        self.standardize = standardize
        # Parámetros 
        self.w_ = None     
        self.mu_ = None    
        self.sigma_ = None 

    def fit(self, X: np.ndarray, y: np.ndarray):
        # TODO: Implementar flujo de entrenamiento
        raise NotImplementedError("TODO")

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: Implementar predicción
        raise NotImplementedError("TODO")

    def _fit_normal(self, X: np.ndarray, y: np.ndarray):
        # TODO: Implementar ecuación normal
        raise NotImplementedError("TODO")

    def _fit_gd(self, X: np.ndarray, y: np.ndarray):
        # TODO: Implementar gradiente descendente
        raise NotImplementedError("TODO")


# Ejecución

def main() -> None:
    args = parse_args()

    X, y, cols = read_csv(args.data, header=args.header, target_idx=args.target_index)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, seed=args.seed)

    model = LinearRegressor(method=args.method, lr=args.lr, epochs=args.epochs, standardize=args.standardize)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"MSE (test): {mse(y_test, y_pred):.6f}")
    print(f"R^2 (test): {r2_score(y_test, y_pred):.6f}")

    if args.predict:
        vec = np.array([float(v.strip()) for v in args.predict.split(",")], dtype=float).reshape(1, -1)
        y_one = model.predict(vec)[0]
        print(f"Predicción para [{args.predict}] -> {y_one:.6f}")

    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
