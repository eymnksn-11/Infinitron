import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# --- 1. Load Dataset ---
print("Fetching MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data / 255.0  # Normalize pixel values to [0, 1]
y = mnist.target.astype(int)

# One-hot encoding for 10 digits
y_onehot = np.zeros((y.size, 10))
y_onehot[np.arange(y.size), y] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.1, random_state=42)

# --- 2. Activation Functions ---
def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

# --- 3. Model Parameters & Initialization ---
input_dim = 784
hidden_dim = 128
output_dim = 10

np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
W2 = np.random.randn(hidden_dim, output_dim) * 0.01

# Hyperparameters
epochs = 10
batch_size = 32
base_scale = 0.001 # Base scaling factor for weight updates

print("-" * 50)

for epoch in range(1, epochs + 1):
    # Shuffle training data
    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    for i in range(0, X_train.shape[0], batch_size):
        x_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        
        # --- Forward Pass ---
        layer_1_in = np.dot(x_batch, W1)
        layer_1_out = sigmoid(layer_1_in)
        
        layer_2_in = np.dot(layer_1_out, W2)
        layer_2_out = sigmoid(layer_2_in)
        
        # --- Dynamic Learning Rate Calculation (Your Equation) ---
        # Based on: d/dx ln(sum f_i)^2
        # Measures the 'gap' in information density via local variation
        local_variation = np.mean(np.square(sigmoid_derivative(layer_2_in)))
        
        # Adaptive log-damped learning rate to fill the information gap
        dynamic_lr = np.log1p(1.0 / (local_variation + 1e-6))
        
        # --- Backpropagation & Grooming ---
        error = y_batch - layer_2_out
        
        # Distribute global error using the logarithmic derivative principle
        grad_layer_2 = error * sigmoid_derivative(layer_2_in)
        grad_layer_1 = grad_layer_2.dot(W2.T) * sigmoid_derivative(layer_1_in)
        
        # --- Weight Updates ---
        # Scaling current update by dynamic_lr to fill the functional gap
        W2 += layer_1_out.T.dot(grad_layer_2) * (dynamic_lr * base_scale)
        W1 += x_batch.T.dot(grad_layer_1) * (dynamic_lr * base_scale)

    # --- End of Epoch Validation ---
    val_l1 = sigmoid(np.dot(X_test, W1))
    val_l2 = sigmoid(np.dot(val_l1, W2))
    
    predictions = np.argmax(val_l2, axis=1)
    labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == labels)
    
    print(f"Epoch {epoch:2d} | Accuracy: {accuracy*100:6.2f}% | LR: {dynamic_lr:8.2f}")

print("-" * 50)
print("Training Completed Successfully.")
