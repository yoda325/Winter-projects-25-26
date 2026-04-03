import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Network Definition

class WirelessNet(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, output_size=64):
        super(WirelessNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def functional_forward(x, weights):
    x = F.linear(x, weights['layer1.weight'], weights['layer1.bias'])
    x = F.relu(x)
    x = F.linear(x, weights['layer2.weight'], weights['layer2.bias'])
    x = F.relu(x)
    x = F.linear(x, weights['layer3.weight'], weights['layer3.bias'])
    return x


# 2. Evaluation Logic

def evaluate_maml():
    print("Loading test dataset and MAML model...")
    data = np.load("wireless_dataset.npz")
    test_x_supp = torch.tensor(data['test_x_support'])
    test_y_supp = torch.tensor(data['test_y_support'])
    test_x_query = torch.tensor(data['test_x_query'])
    test_y_query = torch.tensor(data['test_y_query'])

    num_test_tasks = test_x_supp.shape[0]
    eval_steps = 20  # We evaluate up to 20 steps to get the 20-shot error
    inner_lr = 0.01

    # Load MAML weights
    maml_model = WirelessNet()
    maml_model.load_state_dict(torch.load("meta_model.pth", weights_only=True))

    # Track MAML errors at each step
    maml_errors = np.zeros(eval_steps + 1)

    print(f"Adapting MAML model on {num_test_tasks} unseen test tasks...")

    for task_idx in range(num_test_tasks):
        x_supp, y_supp = test_x_supp[task_idx], test_y_supp[task_idx]
        x_query, y_query = test_x_query[task_idx], test_y_query[task_idx]

        # Start from the smart MAML weights for EACH task
        fast_weights = {name: param for name, param in maml_model.named_parameters()}

        # Step 0 Error (Zero-shot, before any adaptation)
        with torch.no_grad():
            maml_errors[0] += F.mse_loss(functional_forward(x_query, fast_weights), y_query).item()

        # Adapt for up to 20 steps
        for step in range(1, eval_steps + 1):
            preds = functional_forward(x_supp, fast_weights)
            loss = F.mse_loss(preds, y_supp)

            # Compute gradients and update fast_weights
            grads = torch.autograd.grad(loss, fast_weights.values())
            fast_weights = {name: param - inner_lr * grad for ((name, param), grad) in zip(fast_weights.items(), grads)}

            # Evaluate on query set after this step
            with torch.no_grad():
                query_loss = F.mse_loss(functional_forward(x_query, fast_weights), y_query).item()
                maml_errors[step] += query_loss

    # Average the errors across all 20 test tasks
    maml_errors /= num_test_tasks
    return maml_errors


# 3. Generate Plot 2 and Print README Results

if __name__ == "__main__":
    # 1. Get MAML performance
    maml_errors = evaluate_maml()

    # 2. Load the Baseline performance we saved during train.py
    try:
        baseline_errors = np.load("baseline_errors.npy")
    except FileNotFoundError:
        print("Error: baseline_errors.npy not found! Make sure you ran the updated train.py first.")
        exit()

    
    print("\n" + "="*50)
    print(" RESULTS FOR README.MD TABLE")
    print("="*50)
    # The rubric uses dB format. Formula: 10 * log10(MSE)
    base_5_shot = 10 * np.log10(baseline_errors[5])
    base_20_shot = 10 * np.log10(baseline_errors[20])
    maml_5_shot = 10 * np.log10(maml_errors[5])
    maml_20_shot = 10 * np.log10(maml_errors[20])

    print(f"Method                 | 5-shot Error | 20-shot Error")
    print(f"-----------------------------------------------------")
    print(f"Basic model (scratch)  | {base_5_shot:7.2f} dB | {base_20_shot:7.2f} dB")
    print(f"Your MAML model        | {maml_5_shot:7.2f} dB | {maml_20_shot:7.2f} dB")
    print("="*50 + "\n")

    # 4. Generate Plot 2 (Comparing up to 20 steps)
    os.makedirs("results", exist_ok=True)
    steps = range(21) # 0 to 20

    plt.figure(figsize=(8, 5))
    plt.plot(steps, baseline_errors[:21], label="Baseline (Random Init)", color='red', linestyle='--')
    plt.plot(steps, maml_errors[:21], label="MAML (Meta-Trained)", color='green', linewidth=2)

    plt.title("Plot 2: MAML vs Baseline Adaptation on Unseen Tasks")
    plt.xlabel("Number of Adaptation Steps (Shots)")
    plt.ylabel("MSE Error on Query Set")
    plt.xticks(np.arange(0, 21, 5))
    plt.legend()
    plt.grid(True)

    plt.savefig("results/plot_comparison.png")
    print("Plot 2 successfully saved to results/plot_comparison.png.")
