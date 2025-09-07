# mlp-activation-comparison
# MLP Activation Function Comparison

This project builds and evaluates a Multilayer Perceptron (MLP) with three different hidden-layer activation functions: **ReLU**, **LeakyReLU**, and **Sigmoid**.  
The work is based on the code style provided in the textbook repository [Generative Deep Learning (2nd Edition)](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition).

---

## 📂 Project Structure
mlp-activation-comparison/
│
├── mlp_compare.ipynb # Jupyter notebook with code, training runs, and plots
├── mlp_compare.py # (Optional) Python script version
├── requirements.txt # Dependencies
└── README.md # Project description and results


---

## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YOURUSERNAME/mlp-activation-comparison.git
   cd mlp-activation-comparison

2. Install dependences:
   ```bash
   pip install -r requirements.txt
   
3. Run the notebook:
   ```bash
   jupyter notebook mlp_compare.ipynb

----

The notebook will:
Load the Fashion-MNIST dataset (10-class image classification).
Train three identical MLPs, differing only by the hidden-layer activation.
Report training/validation curves and test accuracy.


---


### 📝 Observations

ReLU and LeakyReLU clearly outperform Sigmoid in both convergence speed and final accuracy.
LeakyReLU performs comparably to ReLU and offers stability by allowing a small negative slope, preventing dead neurons.
Sigmoid struggles due to vanishing gradients and saturation, which slows training and lowers accuracy in deeper networks.
For modern deep networks, ReLU-family activations are preferred, while Sigmoid is mainly used in output layers for binary classification.
