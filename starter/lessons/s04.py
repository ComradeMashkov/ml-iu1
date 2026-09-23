"""S4: запуск из starter или starter/notebooks, данные уже находятся на диске."""

# %% Подготовлено: импорты и чтение файла
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data_path = Path("data/sms-spam.tsv")
if not data_path.exists():
    data_path = Path("../data/sms-spam.tsv")
if not data_path.exists():
    data_path = Path("../../data/sms-spam.tsv")
raw_rows = []
for line in data_path.read_text(encoding="utf-8").splitlines():
    label, message = line.split("\t", 1)
    raw_rows.append((label, message))
print("Строк в исходном файле:", len(raw_rows))

# %% 1. Проверка данных и удаление повторов
messages = []
labels = []
seen = set()
for label, message in raw_rows:
    assert label in ["ham", "spam"]
    key = " ".join(message.lower().split())
    assert key != ""
    if key in seen:
        continue
    seen.add(key)
    messages.append(message)
    labels.append(int(label == "spam"))

print("После удаления повторов:", len(messages))
print("Спам:", sum(labels))
print("Обычные:", len(labels) - sum(labels))


# %% Подготовлено: тот же признак, что на S3
def count_words(message):
    count = 0
    for word in message.lower().split():
        word = word.strip(".,!?;:")
        if word in ["free", "win", "prize"]:
            count = count + 1
    return count


# %% 2. Таблица признаков и три части данных
X = []
for message in messages:
    X.append([count_words(message)])
X = np.array(X)
y = np.array(labels)
ids = np.arange(len(y))

train_ids, other_ids = train_test_split(ids, test_size=0.4, random_state=25, stratify=y)
validation_ids, test_ids = train_test_split(
    other_ids, test_size=0.5, random_state=25, stratify=y[other_ids]
)
print("Форма X:", X.shape, "Форма y:", y.shape)
print("Обучение / валидация / тест:", len(train_ids), len(validation_ids), len(test_ids))

# %% 3. Обучение библиотечной модели
model = LogisticRegression(C=float("inf"), max_iter=1000)
model.fit(X[train_ids], y[train_ids])
validation_probability = model.predict_proba(X[validation_ids])[:, 1]
print("Порядок классов:", model.classes_)
print("w =", round(model.coef_[0, 0], 3), "b =", round(model.intercept_[0], 3))
print("Первые пять оценок:", np.round(validation_probability[:5], 3))


# %% 4. Два вида ошибок и точка отсчёта
def count_errors(actual, probability, threshold):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(len(actual)):
        predicted = int(probability[i] >= threshold)
        if actual[i] == 0 and predicted == 1:
            fp = fp + 1
        elif actual[i] == 1 and predicted == 0:
            fn = fn + 1
        elif actual[i] == 1 and predicted == 1:
            tp = tp + 1
        else:
            tn = tn + 1
    return tn, fp, fn, tp


always_zero = np.zeros(len(validation_ids))
baseline = count_errors(y[validation_ids], always_zero, 0.5)
print("Всегда обычное, TN FP FN TP:", baseline)
print("Модель, порог 0.5:", count_errors(y[validation_ids], validation_probability, 0.5))

# %% 5. Выбор порога на валидации
best_cost = float("inf")
best_threshold = None
validation_results = []
for threshold in [0.10, 0.25, 0.50, 0.75]:
    tn, fp, fn, tp = count_errors(y[validation_ids], validation_probability, threshold)
    cost = 5 * fp + fn
    validation_results.append([threshold, tn, fp, fn, tp, cost])
    print("Порог:", threshold, "FP:", fp, "FN:", fn, "Цена:", cost)
    if cost < best_cost:
        best_cost = cost
        best_threshold = threshold

print("Фиксируем порог:", best_threshold)

# %% Подготовлено: реальные ошибки только на валидации
for error_name, true_label, predicted_label in [("FP", 0, 1), ("FN", 1, 0)]:
    for i in range(len(validation_ids)):
        row_id = validation_ids[i]
        prediction = int(validation_probability[i] >= best_threshold)
        if y[row_id] == true_label and prediction == predicted_label:
            print(error_name, "x =", X[row_id, 0], "p =", round(validation_probability[i], 3))
            print(messages[row_id])
            break

# %% 6. Единственная итоговая проверка
test_probability = model.predict_proba(X[test_ids])[:, 1]
tn, fp, fn, tp = count_errors(y[test_ids], test_probability, best_threshold)
accuracy = (tp + tn) / len(test_ids)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
test_cost = 5 * fp + fn
test_baseline_cost = int(sum(y[test_ids]))
print("Тест: TN FP FN TP =", tn, fp, fn, tp)
print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3), "Recall:", round(recall, 3))
print("Цена модели / постоянного ответа:", test_cost, test_baseline_cost)
