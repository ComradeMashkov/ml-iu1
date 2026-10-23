"""Прогноз победы в Counter-Strike. Run from starter with a clean Python process."""

# S08-C01
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
torch.manual_seed(8)

# S08-C02
folder = Path.cwd()
if folder.name == "solutions":
    folder = folder.parent
if folder.name == "notebooks":
    folder = folder.parent
if folder.name != "starter":
    folder = folder / "starter"
with (folder / "data/csgo-match-features.csv").open() as file:
    rows = list(csv.DictReader(file))
print("Матчей в файле:", len(rows))
for example_index, row in enumerate(rows):
    if row["match_id"] == "2332375":
        break
example = rows[example_index]
print(example["team1"], "против", example["team2"])
print("Предыдущие победы из 10:", example["wins1_last10"], example["wins2_last10"])

# S08-C03
features = []
targets = []
for row in rows:
    features.append([int(row["wins1_last10"]) / 10, int(row["wins2_last10"]) / 10])
    targets.append(int(row["team1_won"]))
X = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
print(X.shape, y.shape)
print("Вход DETONA - Denial:", features[example_index])

# S08-C04
train_ids = [i for i, row in enumerate(rows) if row["split"] == "train"]
validation_ids = [i for i, row in enumerate(rows) if row["split"] == "validation"]
test_ids = [i for i, row in enumerate(rows) if row["split"] == "test"]
X_train, y_train = X[train_ids], y[train_ids]
X_validation, y_validation = X[validation_ids], y[validation_ids]
X_test, y_test = X[test_ids], y[test_ids]
test_rows = [rows[i] for i in test_ids]
print("train / validation / test:", len(train_ids), len(validation_ids), len(test_ids))

# S08-C05
baseline_probability = y_train.mean().item()
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
print("Постоянная оценка:", round(baseline_probability, 4))
print("Параметров:", sum(parameter.numel() for parameter in model.parameters()))

# S08-C06
history = []
for step in range(500):
    optimizer.zero_grad()
    logits = model(X_train)
    loss = loss_function(logits, y_train)
    loss.backward()
    optimizer.step()
    history.append(loss.item())
    if step in [0, 99, 499]:
        print(step + 1, round(loss.item(), 6))
plt.plot(range(1, 501), history)
plt.xlabel("Вычисление потери до обновления")
plt.ylabel("Средняя логарифмическая потеря на 6218 train")
plt.show()

# S08-C07
shares = torch.arange(11) / 10
grid = torch.cartesian_prod(shares, shares)
with torch.no_grad():
    hidden = torch.relu(model[0](grid))
    surface = torch.sigmoid(model(grid))
plt.figure(figsize=(12, 5))
for i in range(9):
    plt.subplot(2, 5, i + 1)
    values = hidden[:, i] if i < 8 else surface[:, 0]
    plt.imshow(
        values.reshape(11, 11), origin="lower", extent=[-0.05, 1.05, -0.05, 1.05], cmap="viridis"
    )
    plt.scatter([0.6], [0.8], color="red")
    plt.title("нейрон " + str(i + 1) if i < 8 else "итог p")
plt.tight_layout()
plt.show()
print("p для [0.8, 0.6]:", round(surface[8 * 11 + 6, 0].item(), 4))

# S08-C08
example_actual = np.array([1, 0])
forecast_60 = np.array([0.6, 0.6])
forecast_90 = np.array([0.9, 0.9])
print(
    "Учебные 2 матча, прогнозы 60%:", round(float(((forecast_60 - example_actual) ** 2).mean()), 2)
)
print(
    "Учебные 2 матча, прогнозы 90%:", round(float(((forecast_90 - example_actual) ** 2).mean()), 2)
)

with torch.no_grad():
    validation_p = torch.sigmoid(model(X_validation)).flatten().numpy()
validation_actual = y_validation.flatten().numpy()
validation_squared = (validation_p - validation_actual) ** 2
print("validation n:", len(validation_p))
print("Сумма квадратов:", float(validation_squared.sum()))
print("Brier сети:", float(validation_squared.mean()))
print("Brier постоянной оценки:", float(((baseline_probability - validation_actual) ** 2).mean()))

# S08-C09
stake = 1.0
initial_bank = 1000.0
minimum_expected_profit = 0.05
example_p = 0.6
example_odds = 1.5
print("Ожидание на единицу по выбранной p:", round(example_p * example_odds - 1, 3))

# S08-C10
with torch.no_grad():
    test_p = torch.sigmoid(model(X_test)).flatten().numpy()
actual = y_test.flatten().numpy()
squared = (test_p - actual) ** 2
correct = int(((test_p >= 0.5) == actual).sum())
print("Верных исходов:", correct, "/", len(test_p))
print("Если всегда называть первую команду:", int(actual.sum()), "/", len(actual))
print("Сумма квадратов:", float(squared.sum()))
print("Brier сети:", float(squared.mean()))
print("Brier постоянной оценки:", float(((baseline_probability - actual) ** 2).mean()))

# S08-C11
quoted_rows = []
quoted_p = []
for position, row in enumerate(test_rows):
    if row["odds1"]:
        quoted_rows.append(row)
        quoted_p.append(test_p[position])
print("Матчей с коэффициентами:", len(quoted_rows))
for position in [0, 6]:
    example = quoted_rows[position]
    p1 = float(quoted_p[position])
    p2 = 1 - p1
    odds = float(example["odds1"])
    expected_profit = p1 * odds - 1
    print(example["team1"], "против", example["team2"])
    print("Оценка победы", example["team1"], ":", round(p1 * 100, 2), "%")
    print("Оценка победы", example["team2"], ":", round(p2 * 100, 2), "%")
    print("Коэффициент на", example["team1"], ":", odds)
    print("Ожидаемый чистый результат на 1:", round(expected_profit, 4))
    if expected_profit > minimum_expected_profit:
        print("Решение: ставка 1 на", example["team1"])
    else:
        print("Решение: пропуск")

# S08-C12
bank = initial_bank
bank_history = [bank]
bet_count = 0
bet_wins = 0
for row, probability in zip(quoted_rows, quoted_p):
    odds = float(row["odds1"])
    expected_profit = float(probability) * odds - 1
    if expected_profit > minimum_expected_profit:
        outcome = int(row["team1_won"])
        profit = stake * (odds * outcome - 1)
        bank = bank + profit
        bet_count = bet_count + 1
        bet_wins = bet_wins + outcome
        if bet_count <= 3:
            print("Ставка на", row["team1"], "против", row["team2"])
            print(
                "Оценка победы:",
                round(float(probability) * 100, 2),
                "%",
                "чистый итог:",
                round(profit, 2),
            )
    bank_history.append(bank)
print("Ставок:", bet_count, "выиграно:", bet_wins)
print("Чистый итог:", round(bank - initial_bank, 2))
if bet_count > 0:
    print("Итог / сумма ставок:", round((bank - initial_bank) / (stake * bet_count), 4))

# S08-C13
plt.figure(figsize=(8, 4))
plt.plot(bank_history)
plt.axhline(initial_bank, color="gray", linestyle="--")
plt.xlabel("Рассмотрено матчей с исторической ценой")
plt.ylabel("Накопленный условный итог")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# S08-C14
market_p = []
quoted_actual = []
for row in quoted_rows:
    inverse1 = 1 / float(row["odds1"])
    inverse2 = 1 / float(row["odds2"])
    market_p.append(inverse1 / (inverse1 + inverse2))
    quoted_actual.append(int(row["team1_won"]))
market_p = np.array(market_p)
quoted_actual = np.array(quoted_actual)
print("Одинаковых матчей:", len(quoted_actual))
print("Brier сети:", float(((np.array(quoted_p) - quoted_actual) ** 2).mean()))
print("Brier оценки из цен:", float(((market_p - quoted_actual) ** 2).mean()))
print(
    "Угадано сетью:",
    int(((np.array(quoted_p) >= 0.5) == quoted_actual).sum()),
    "/",
    len(quoted_actual),
)
print(
    "Угадано фаворитом букмекера:",
    int(((market_p >= 0.5) == quoted_actual).sum()),
    "/",
    len(quoted_actual),
)
