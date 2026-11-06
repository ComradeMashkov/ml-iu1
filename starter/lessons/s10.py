"""Рэп, рок или поп: деревья и лес. Run from starter with a clean Python process."""

# S10-C01
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

folder = Path.cwd()
if folder.name == "solutions":
    folder = folder.parent
if folder.name == "notebooks":
    folder = folder.parent
if folder.name != "starter":
    folder = folder / "starter"
with (folder / "data/russian-genres.csv").open(encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
print("Песен:", len(rows))
print(next(row for row in rows if row["track"] == "Что такое осень"))

# S10-C02
names = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "valence",
    "tempo",
    "minutes",
]
X = np.array([[float(row[name]) for name in names] for row in rows])
y = np.array([row["genre"] for row in rows])
part = np.array([row["split"] for row in rows])
train, validation, test, usa = [part == name for name in ["train", "validation", "test", "usa"]]
print("train, validation, test, usa:", train.sum(), validation.sum(), test.sum(), usa.sum())
print("Всегда рок на validation:", round((y[validation] == "рок").mean(), 3))

# S10-C03
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X[train], y[train])
print(export_text(tree, feature_names=names))
print("train:", round(tree.score(X[train], y[train]), 3))
print("validation:", round(tree.score(X[validation], y[validation]), 3))

# S10-C04
free_tree = DecisionTreeClassifier(random_state=0)
free_tree.fit(X[train], y[train])
print("Листьев:", free_tree.get_n_leaves())
print("train:", round(free_tree.score(X[train], y[train]), 3))
print("validation:", round(free_tree.score(X[validation], y[validation]), 3))

# S10-C05
rng = np.random.default_rng(10)
ids = rng.permutation(np.where(train)[0])
half = len(ids) // 2
first = DecisionTreeClassifier(random_state=0).fit(X[ids[:half]], y[ids[:half]])
second = DecisionTreeClassifier(random_state=0).fit(X[ids[half:]], y[ids[half:]])
print(
    "Спорят на доле песен:",
    round((first.predict(X[validation]) != second.predict(X[validation])).mean(), 3),
)

# S10-C06
forest = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
forest.fit(X[train], y[train])
models = {"дерево из 3 вопросов": tree, "дерево наизусть": free_tree, "лес": forest}
for name, model in models.items():
    print(
        name,
        round(model.score(X[validation], y[validation]), 3),
        round(model.score(X[test], y[test]), 3),
    )
print("всегда рок", round((y[validation] == "рок").mean(), 3), round((y[test] == "рок").mean(), 3))

# S10-C07
songs = ["Life", "чайлдфри", "Город под подошвой", "Что такое осень", "KARI", "молчи"]
for track in songs:
    index = next(
        i for i, row in enumerate(rows) if row["track"] == track and row["split"] == "test"
    )
    votes = forest.predict_proba(X[[index]])[0]
    print(track, rows[index]["genre"], dict(zip(forest.classes_.tolist(), votes.round(2).tolist())))

# S10-C08
answers = forest.predict(X[usa])
print("Назвал рэпом:", round((answers == "рэп").mean(), 3))
print({genre: int((answers == genre).sum()) for genre in forest.classes_.tolist()})
