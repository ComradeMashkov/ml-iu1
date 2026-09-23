"""S2: turn a small, deliberately imperfect CSV into the S1 feature table."""

# %% Подготовлено: чтение CSV и папка результата
import csv
from pathlib import Path

for folder in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent, Path.cwd() / "starter"]:
    if (folder / "data" / "intro-temperature.csv").is_file():
        starter_folder = folder
        break
else:
    raise FileNotFoundError("Откройте notebook из starter или запустите код из папки курса")

with (starter_folder / "data" / "intro-temperature.csv").open(encoding="utf-8") as file:
    raw_rows = list(csv.DictReader(file))

# %% 1. Прочитать одну строку
first_row = raw_rows[0]
print(first_row)
print("Номер запуска:", first_row["run_id"])
print("Первое измерение:", first_row["t1_c"])
print("Тип значения:", type(first_row["t1_c"]))
print("Строк в файле:", len(raw_rows))

# %% 2. Проверить метки и измерения
valid_rows = []
rejected = []

for row in raw_rows:
    run_id = row["run_id"]
    if row["inspection"] not in ["0", "1"]:
        rejected.append([run_id, "неизвестная метка"])
        continue

    if row["t1_c"] == "" or row["t2_c"] == "" or row["t3_c"] == "":
        rejected.append([run_id, "пропущено измерение"])
        continue

    try:
        t1 = float(row["t1_c"])
        t2 = float(row["t2_c"])
        t3 = float(row["t3_c"])
    except ValueError:
        rejected.append([run_id, "температура не является числом"])
        continue

    target = int(row["inspection"])
    valid_rows.append([run_id, t1, t2, t3, target])

print("Отклонённые строки:", rejected)
print("Строк после проверки:", len(valid_rows))

# %% 3. Удалить точный повтор
unique_rows = []
duplicates = 0

for row in valid_rows:
    if row in unique_rows:
        duplicates = duplicates + 1
    else:
        unique_rows.append(row)

print("Точных повторов:", duplicates)
print("Уникальных строк:", len(unique_rows))

# %% 4. Из трёх измерений получить один признак
run_ids = []
temperature = []
target = []

for row in unique_rows:
    run_id = row[0]
    mean_temperature = (row[1] + row[2] + row[3]) / 3
    inspection = row[4]

    run_ids.append(run_id)
    temperature.append(mean_temperature)
    target.append(inspection)

print("Температуры:", temperature)
print("Метки:", target)

# %% 5. Проверить связь признака с исходной строкой
for i in range(len(run_ids)):
    print(run_ids[i], "средняя температура:", temperature[i], "метка:", target[i])

first_measurements = unique_rows[0][1:4]
print("F01, исходные измерения:", first_measurements)
print("F01, среднее:", sum(first_measurements) / len(first_measurements))

# %% 6. Сохранить понятную таблицу
output_folder = starter_folder / "reports"
output_folder.mkdir(exist_ok=True)
output_path = output_folder / "s2-temperature-features.csv"

with output_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["run_id", "mean_temperature_c", "inspection"])
    for i in range(len(run_ids)):
        writer.writerow([run_ids[i], temperature[i], target[i]])

print("Сохранено: reports/s2-temperature-features.csv")
