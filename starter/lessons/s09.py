"""Обвинение детектора. Run from starter with a clean Python process."""

# S09-C01
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

folder = Path.cwd()
if folder.name == "solutions":
    folder = folder.parent
if folder.name == "notebooks":
    folder = folder.parent
if folder.name != "starter":
    folder = folder / "starter"
with (folder / "data/coat-texts.csv").open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
print("Текстов:", len(rows))
print(next(row["text"] for row in rows if "старостой" in row["text"]))

# S09-C02
names = ["word_length", "unique_share", "personal_per_100", "commas_per_100"]
X = np.array([[float(row[name]) for name in names] for row in rows])
y = np.array([int(row["ai"]) for row in rows])
part = np.array([row["split"] for row in rows])
train, test = part == "train", part == "test"
model = LogisticRegression(max_iter=2000)
model.fit(X[train], y[train])
scores = model.predict_proba(X)[:, 1]
print(dict(zip(names, model.coef_[0].round(2).tolist())))
print("Доля верных на test:", ((scores[test] >= 0.5) == y[test]).mean())

# S09-C03
humans = test & (y == 0)
machines = test & (y == 1)
caught = (scores[machines] >= 0.5).mean()
false_alarm = (scores[humans] >= 0.5).mean()
print("Находит нейротекстов:", caught)
print("Обвиняет людей:", false_alarm)


# S09-C04
def share_guilty(base, caught, false_alarm):
    guilty = base * caught
    honest = (1 - base) * false_alarm
    return guilty / (guilty + honest)


print("Реклама, 5%:", round(share_guilty(0.05, 0.98, 0.02), 3))
print("Реклама, 1%:", round(share_guilty(0.01, 0.98, 0.02), 3))
print("Наш детектор, 5%:", round(share_guilty(0.05, caught, false_alarm), 3))

# S09-C05
for threshold in [0.5, 0.7, 0.9]:
    found = 50 * (scores[machines] >= threshold).mean()
    wrong = 950 * (scores[humans] >= threshold).mean()
    print(threshold, "поймано:", round(found, 1), "честных обвинено:", round(wrong, 1))

# S09-C06
personal = X[:, 2] > 0
print("С личными словами:", humans[personal].sum(), (scores[humans & personal] >= 0.5).mean())
print("Без личных слов:", humans[~personal].sum(), (scores[humans & ~personal] >= 0.5).mean())
