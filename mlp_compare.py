import os, random, numpy as np, tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd

# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
num_classes = 10

x_train = (x_train.astype("float32") / 255.0).reshape(-1, 28*28)
x_test  = (x_test.astype("float32") / 255.0).reshape(-1, 28*28)
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test, num_classes)

VAL_SPLIT = 0.1

def build_mlp(hidden_activation: str):
    model = Sequential()
    model.add(Input(shape=(784,)))

    def block(units):
        if hidden_activation.lower() == "leakyrelu":
            model.add(Dense(units))
            model.add(LeakyReLU(alpha=0.01))
        else:
            model.add(Dense(units, activation=hidden_activation))
        model.add(Dropout(0.2))

    block(256); block(128); block(64)
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

histories = {}
results = []

for act in ["relu", "leakyrelu", "sigmoid"]:
    print(f"\n=== Training with {act.upper()} ===")
    model = build_mlp(act)
    history = model.fit(x_train, y_train,
                        validation_split=VAL_SPLIT,
                        epochs=5, batch_size=128, verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    histories[act] = history
    results.append({"activation": act, "test_acc": float(test_acc), "test_loss": float(test_loss)})

df = pd.DataFrame(results)
print("\n=== Test Accuracy Comparison ===")
print(df.to_string(index=False))

