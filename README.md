# Infinitron
In the world of standard Backpropagation (BP), every layer attempts to navigate an infinite sea of possibilities through the chained multiplication of derivatives. Infinitron aims to find that same path but through a fundamentally different philosophy.

# Key Principles
# Binary Reduction of Possibilities 
Instead of traditional gradient flow, Infinitron evaluates activation functions in sequential pairs. By taking the logarithmic derivative of their mean, it identifies the "correct" activation path among infinite variations. This process effectively evaporates half of the remaining "probability sea" with each iteration, leading to the rapid convergence observed in tests.

# Dynamic Gap Filling 
The differences between functions are treated as "functional gaps" within current probabilities. Rather than using a static Learning Rate (LR), Infinitron calculates an instantaneous LR based on the magnitude of these gaps. It doesn't just step; it fills the void.

# Additive Gradient Hierarchy
The layer hierarchy transcends the classical Chain Rule. Derivatives are not multiplied; instead, they are summed using logarithmic identities.

# Computational Efficiency
By utilizing the global error margin (calculated from the left side of the equation), individual layer derivatives can simply be "subtracted." This eliminates the need for heavy BP caching and significantly streamlines the computational workload.

# Benchmarks

<img width="375" height="208" alt="mnist-results" src="https://github.com/user-attachments/assets/35c0b45c-ce77-4b10-b984-87f27b1c610c" />

<img width="386" height="222" alt="xor-results" src="https://github.com/user-attachments/assets/08b26f65-74fb-48a6-885f-42dd52128f75" />

# Main Equation of Algorithm
<img width="1014" height="214" alt="equation" src="https://github.com/user-attachments/assets/5b1b8807-a05f-44fd-b4d3-4429e6742f6c" />

