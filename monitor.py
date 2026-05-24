import os

if os.path.exists("model.pkl"):
    print("Model is available")
else:
    print("Model not found")