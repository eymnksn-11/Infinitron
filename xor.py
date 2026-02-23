import numpy as np

# --- 1. Activation Functions ---
def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

# --- 2. Dataset (XOR) ---
# Input features
X = np.array([[0,0], [0,1], [1,0], [1,1]])
# Target outputs
y = np.array([[0], [1], [1], [0]])

# --- 3. Model Parameters & Initialization ---
# Setting seed for reproducibility
np.random.seed(42)

# Dimensions: 2 inputs -> 4 hidden neurons -> 1 output
input_dim = 2
hidden_dim = 4
output_dim = 1

# Random weight initialization
W1 = np.random.randn(input_dim, hidden_dim) 
W2 = np.random.randn(hidden_dim, output_dim)

# Hyperparameters
epochs = 500
base_scale = 0.1  # Scaling factor for the dynamic update

print("-" * 50)
for epoch in range(1, epochs + 1):
    # --- Forward Pass ---
    layer_1_in = np.dot(X, W1)
    layer_1_out = sigmoid(layer_1_in)
    
    layer_2_in = np.dot(layer_1_out, W2)
    layer_2_out = sigmoid(layer_2_in)
    
    # --- Dynamic Learning Rate Calculation (The Core Theory) ---
    # Based on: (d/dx ln(sum f_i))^2
    # Measures 'local variation' to detect gaps in the information space
    local_variation = np.mean(np.square(sigmoid_derivative(layer_2_in)))
    
    # Gap-Filling LR: As variation decreases, step size expands to reach convergence
    dynamic_lr = 1.0 / (local_variation + 1e-6)
    
    # --- Backpropagation & Error Distribution ---
    error = y - layer_2_out
    
    # Logarithmic derivative-based error distribution
    grad_layer_2 = error * sigmoid_derivative(layer_2_in)
    grad_layer_1 = grad_layer_2.dot(W2.T) * sigmoid_derivative(layer_1_in)
    
    # --- Weight Updates ---
    # Applying dynamic_lr to groom layers according to hierarchical gaps
    W2 += layer_1_out.T.dot(grad_layer_2) * (dynamic_lr * base_scale)
    W1 += X.T.dot(grad_layer_1) * (dynamic_lr * base_scale)
    
    # Log progress every 100 epochs
    if epoch % 100 == 0 or epoch == 1:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Dynamic LR: {dynamic_lr:8.2f}")

# --- 4. Final Inference ---
print("-" * 50)
print("Final Predictions:")
print(np.round(layer_2_out, 3))
