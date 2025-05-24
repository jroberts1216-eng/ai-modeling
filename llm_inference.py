from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load a small LLM from Hugging Face
model_name = "tiiuae/falcon-rw-1b"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Simple prompt
prompt = "Explain the difference between AI and machine learning."

# Tokenize and generate response
inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=100)

# Decode and print result
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated response:\n")
print(response)
