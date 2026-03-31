import pickle
import matplotlib.pyplot as plt


with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]

plt.plot(model.losses)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("loss_curve.png", bbox_inches="tight")
print("Loss curve saved as loss_curve.png")
