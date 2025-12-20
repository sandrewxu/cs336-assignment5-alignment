
import torch
import torch.nn as nn
from cs336_alignment.functional import get_response_log_probs
from cs336_alignment.sft import sft_microbatch_train_step

# Mock model
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
    
    def forward(self, input_ids):
        # Return dummy logits
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, 10, requires_grad=True)
        return type('Output', (), {'logits': logits})()

model = MockModel()
input_ids = torch.randint(0, 10, (2, 5))
labels = torch.randint(0, 10, (2, 5))
response_mask = torch.ones((2, 5))

print("--- Testing get_response_log_probs gradients ---")
try:
    results = get_response_log_probs(model, input_ids, labels)
    log_probs = results['log_probs']
    print(f"Log probs requires_grad: {log_probs.requires_grad}")
    if not log_probs.requires_grad:
        print("ISSUE CONFIRMED: Log probs do not have gradients (likely due to inference_mode).")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Testing sft_microbatch_train_step type handling ---")
try:
    # Intentionally passing the dict as sft.py does
    loss, metadata = sft_microbatch_train_step(results, response_mask, 1)
    print("Function accepted dict.")
except Exception as e:
    print(f"ISSUE CONFIRMED: Function failed with dict input. Error: {e}")

print("\n--- Testing sft_microbatch_train_step optimization direction ---")
# Manually create tensor with grad for this test
log_probs_tensor = torch.tensor([[-2.0]], requires_grad=True) # Low probability
mask = torch.ones_like(log_probs_tensor)
loss, _ = sft_microbatch_train_step(log_probs_tensor, mask, 1)
print(f"Input log probs: {log_probs_tensor.data}")
print(f"Calculated loss: {loss}")

print("Checking internal backward behavior manually...")
# Since sft_microbatch_train_step calls .backward() internally, we check the grad of the input tensor
print(f"Gradient on log_probs: {log_probs_tensor.grad}")
if log_probs_tensor.grad is not None and log_probs_tensor.grad > 0:
    print("Gradient is positive. Update rule: theta = theta - lr * grad.")
    print("New log_probs would be: -2.0 - lr * (positive) = more negative.")
    print("ISSUE CONFIRMED: Minimizing this loss decreases the log probability.")
elif log_probs_tensor.grad is None:
    print("Gradient is None.")
else:
    print(f"Gradient is {log_probs_tensor.grad}")
