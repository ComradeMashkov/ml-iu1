"""Учим сеть узнавать рисунки. Run from starter with a clean Python process."""

# S07-C01
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

torch.set_num_threads(1)
_ = torch.manual_seed(7)

# S07-C02
folder = Path.cwd()
if folder.name == "solutions":
    folder = folder.parent
if folder.name == "notebooks":
    folder = folder.parent
if folder.name != "starter":
    folder = folder / "starter"
data = np.load(folder / "data/quickdraw-cat-teapot.npz")
images = data["images"]
labels = data["labels"]
names = ["кот", "чайник"]
print(images.shape, labels.shape)

# S07-C03
examples = [3, 4, 5, 603, 604, 605]
plt.figure(figsize=(9, 5))
for position, index in enumerate(examples):
    plt.subplot(2, 3, position + 1)
    plt.imshow(images[index].reshape(28, 28), cmap="gray_r")
    plt.title(names[labels[index]])
    plt.axis("off")
plt.tight_layout()
plt.show()

# S07-C04
w = torch.tensor(1.0, requires_grad=True)
h = torch.relu(w)
z = h - 1
one_loss = nn.BCEWithLogitsLoss()(z, torch.tensor(1.0))
one_loss.backward()
print("loss =", round(one_loss.item(), 6))
print("gradient =", w.grad.item())

# S07-C05
negative_w = torch.tensor(-1.0, requires_grad=True)
negative_h = torch.relu(negative_w)
negative_z = negative_h - 1
negative_loss = nn.BCEWithLogitsLoss()(negative_z, torch.tensor(1.0))
negative_loss.backward()
print("negative loss =", round(negative_loss.item(), 6))
print("negative gradient =", negative_w.grad.item())

# S07-C06
X = images / 255.0
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=1 / 3, random_state=7, stratify=labels
)
print("train:", len(X_train), "test:", len(X_test))

# S07-C07
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
print(X_train.shape, y_train.shape)

# S07-C08
torch.manual_seed(7)
model = nn.Sequential(nn.Linear(784, 16), nn.ReLU(), nn.Linear(16, 1))
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
print(sum(parameter.numel() for parameter in model.parameters()))

# S07-C09
history = []
test_history = []
for step in range(200):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = loss_function(logits, y_train)
    with torch.no_grad():
        test_history.append(loss_function(model(X_test), y_test).item())
    loss.backward()
    optimizer.step()
    history.append(loss.item())
    if step in [0, 49, 99, 199]:
        print(step + 1, round(loss.item(), 6), round(test_history[-1], 6))

# S07-C10
plt.figure(figsize=(7, 4))
plt.plot(range(1, 201), history, label="800 train: по ним учим")
plt.plot(range(1, 201), test_history, label="400 test: только смотрим")
plt.xlabel("Шаг обучения")
plt.ylabel("Средняя BCE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
best_step = test_history.index(min(test_history)) + 1
print("Наименьшая test-потеря:", round(min(test_history), 4), "на шаге", best_step)

# S07-C11
with torch.no_grad():
    train_answers = (model(X_train) >= 0).int()
    test_answers = (model(X_test) >= 0).int()
train_errors = int((train_answers != y_train).sum())
test_errors = int((test_answers != y_test).sum())
print("train:", train_errors, "/", len(y_train))
print("test:", test_errors, "/", len(y_test))
print("test error:", test_errors / len(y_test))

# S07-C12
mistakes = torch.where((test_answers != y_test).flatten())[0]
plt.figure(figsize=(9, 5))
for position, index in enumerate(mistakes[:6]):
    plt.subplot(2, 3, position + 1)
    plt.imshow(X_test[index].reshape(28, 28), cmap="gray_r")
    actual = names[int(y_test[index].item())]
    predicted = names[int(test_answers[index].item())]
    plt.title(actual + " / ответ: " + predicted)
    plt.axis("off")
plt.tight_layout()
plt.show()

# S07-C13
torch.manual_seed(7)
linear = nn.Linear(784, 1)
linear_optimizer = torch.optim.SGD(linear.parameters(), lr=0.5)
for step in range(200):
    linear_optimizer.zero_grad()
    linear_loss = loss_function(linear(X_train), y_train)
    linear_loss.backward()
    linear_optimizer.step()
with torch.no_grad():
    linear_train = int(((linear(X_train) >= 0).int() != y_train).sum())
    linear_test = int(((linear(X_test) >= 0).int() != y_test).sum())
print("Без скрытого слоя:  train", linear_train, "/ 800, test", linear_test, "/ 400")
print("16 нейронов с ReLU: train", train_errors, "/ 800, test", test_errors, "/ 400")

# S07-C14
picture = X_test[0]
with torch.no_grad():
    hidden = torch.relu(model[0](picture))
    votes = hidden * model[2].weight[0]
    z = votes.sum() + model[2].bias[0]
print("Метка:", names[int(y_test[0].item())], "| p(чайник) =", round(torch.sigmoid(z).item(), 3))
plt.figure(figsize=(12, 3.5))
plt.subplot(2, 9, 1)
plt.imshow(picture.reshape(28, 28), cmap="gray_r")
plt.axis("off")
for i in range(16):
    plt.subplot(2, 9, i + 2 + i // 8)
    plt.imshow(model[0].weight[i].detach().reshape(28, 28), cmap="bwr", vmin=-0.25, vmax=0.25)
    plt.title(round(votes[i].item(), 1))
    plt.axis("off")
plt.show()

# S07-C15
rabbit = torch.tensor(data["rabbit"] / 255.0, dtype=torch.float32).reshape(1, 784)
with torch.no_grad():
    teapot_score = torch.sigmoid(model(rabbit)).item()
print("Оценка чайника:", round(teapot_score, 3))
plt.imshow(rabbit.reshape(28, 28), cmap="gray_r")
plt.title("Исходная категория: кролик")
plt.axis("off")
plt.show()
