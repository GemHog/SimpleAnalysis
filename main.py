import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_VALUE = -10000
MAX_VALUE = 10000
SOURCE_FILE = "source_data.csv"
RESULT_FILE = "result.txt"
DATAFRAME_FILE = "dataframe_result.csv"


def get_count() -> int:
    while True:
        try:
            count = int(input("Введите количество элементов (1-10000): "))
            if 1 <= count <= 10000:
                return count
            print("Ошибка: число должно быть в диапазоне от 1 до 10000.")
        except ValueError:
            print("Ошибка: необходимо ввести целое число.")

def generate_dataset(data_size: int) -> None:
    rng = np.random.default_rng()

    data = rng.integers(
        low=MIN_VALUE,
        high=MAX_VALUE + 1,
        size=data_size
    )

    dataframe = pd.DataFrame({"value": data})
    dataframe.to_csv(SOURCE_FILE, index=False, encoding="utf-8")

    print(f"Файл {SOURCE_FILE} успешно создан.")

def load_series() -> pd.Series:
    dataframe = pd.read_csv(SOURCE_FILE)

    series = pd.to_numeric(dataframe["value"], errors="coerce")

    return series


def clean_series(series: pd.Series) -> pd.Series:
    cleaned_series = series.dropna()

    cleaned_series = cleaned_series[
        (cleaned_series >= MIN_VALUE) & (cleaned_series <= MAX_VALUE)
    ]

    cleaned_series = cleaned_series.astype(int)

    return cleaned_series

def round_to_hundreds(value: int) -> int:
    if value >= 0:
        return int(math.floor(value / 100 + 0.5) * 100)

    return int(math.ceil(value / 100 - 0.5) * 100)

def calculate_statistics(series: pd.Series) -> dict:
    mean_value = series.sum() / len(series)

    standard_deviation = math.sqrt(
        sum((value - mean_value) ** 2 for value in series) / len(series)
    )

    value_counts = series.value_counts()
    repeated_unique_values = value_counts[value_counts > 1].count()
    duplicated_elements = series.duplicated().sum()

    statistics = {
        "Количество элементов": len(series),
        "Минимальное значение": series.min(),
        "Количество значений, которые повторяются": repeated_unique_values,
        "Количество повторяющихся элементов": duplicated_elements,
        "Максимальное значение": series.max(),
        "Сумма чисел": series.sum(),
        "Среднее значение": mean_value,
        "Среднеквадратическое отклонение": standard_deviation,
    }

    return statistics

def define_sign_group(value: int) -> str:
    if value < 0:
        return "Отрицательное значение"

    if value > 0:
        return "Положительное значение"

    return "Ноль"

def create_analysis_dataframe(series: pd.Series) -> pd.DataFrame:
    dataframe = pd.DataFrame({
        "Исходные значения": series.reset_index(drop=True)
    })

    dataframe["Сортировка по возрастанию"] = pd.Series(
        sorted(series)
    )

    dataframe["Сортировка по убыванию"] = pd.Series(
        sorted(series, reverse=True)
    )

    dataframe["Округление до сотен"] = dataframe["Исходные значения"].apply(
        round_to_hundreds
    )

    dataframe["Группа по знаку"] = dataframe["Исходные значения"].apply(
        define_sign_group
    )

    return dataframe

def calculate_range_sums(dataframe: pd.DataFrame) -> pd.Series:
    range_sums = dataframe.groupby("Округление до сотен")[
        "Исходные значения"
    ].sum()

    return range_sums

def get_dataframe_info(dataframe: pd.DataFrame) -> str:
    buffer = io.StringIO()
    dataframe.info(buf=buffer)

    return buffer.getvalue()

def save_results(
    statistics: dict,
    dataframe: pd.DataFrame,
    range_sums: pd.Series
) -> None:
    dataframe_info = get_dataframe_info(dataframe)

    result_lines = [
        "Результаты выполнения индивидуального задания SimpleAnalysis",
        "",
        "1. Стандартные числовые характеристики:",
    ]

    for name, value in statistics.items():
        result_lines.append(f"{name}: {value}")

    result_lines.extend([
        "",
        "2. Сводная информация о DataFrame:",
        dataframe_info,
        "",
        "3. Количество элементов по группам знака:",
        str(dataframe["Группа по знаку"].value_counts()),
        "",
        "4. Суммарные значения по диапазонам округления до сотен:",
        str(range_sums),
    ])

    result_text = "\n".join(result_lines)

    print(result_text)

    with open(RESULT_FILE, "w", encoding="utf-8") as file:
        file.write(result_text)

    dataframe.to_csv(DATAFRAME_FILE, index=False, encoding="utf-8")

    print(f"\nРезультаты сохранены в файл {RESULT_FILE}.")
    print(f"DataFrame сохранен в файл {DATAFRAME_FILE}.")

def plot_line(series: pd.Series) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(series.values)

    plt.title("Линейный график исходных значений")
    plt.xlabel("Порядковый номер элемента")
    plt.ylabel("Значение")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_1_line.png", dpi=300)
    plt.show()

def plot_histogram(dataframe: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.hist(
        dataframe["Округление до сотен"],
        bins=30,
        edgecolor="black"
    )

    plt.title("Гистограмма значений, округленных до сотен")
    plt.xlabel("Округленные значения")
    plt.ylabel("Количество элементов")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_2_histogram.png", dpi=300)
    plt.show()

def plot_sorted_values(dataframe: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(
        dataframe["Сортировка по возрастанию"].values,
        label="Сортировка по возрастанию"
    )

    plt.plot(
        dataframe["Сортировка по убыванию"].values,
        label="Сортировка по убыванию"
    )

    plt.title("Сравнение отсортированных значений")
    plt.xlabel("Порядковый номер элемента")
    plt.ylabel("Значение")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_3_sorted.png", dpi=300)
    plt.show()

def plot_range_sums(range_sums: pd.Series) -> None:
    plt.figure(figsize=(12, 6))

    plt.plot(
        range_sums.index,
        range_sums.values,
        marker="o"
    )

    plt.title("Суммарные значения по диапазонам округления")
    plt.xlabel("Диапазон округления до сотен")
    plt.ylabel("Сумма исходных значений")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_4_range_sums.png", dpi=300)
    plt.show()

def create_plots(series: pd.Series, dataframe: pd.DataFrame, range_sums: pd.Series) -> None:
    plot_line(series)
    plot_histogram(dataframe)
    plot_sorted_values(dataframe)
    plot_range_sums(range_sums)

def main() -> None:
    data_size = get_count()

    generate_dataset(data_size)

    source_series = load_series()
    clean_data = clean_series(source_series)

    print("\nПервые 10 элементов набора:")
    print(clean_data.head(10))

    statistics = calculate_statistics(clean_data)

    analysis_dataframe = create_analysis_dataframe(clean_data)

    range_sums = calculate_range_sums(analysis_dataframe)

    save_results(
        statistics=statistics,
        dataframe=analysis_dataframe,
        range_sums=range_sums
    )

    create_plots(
        series=clean_data,
        dataframe=analysis_dataframe,
        range_sums=range_sums
    )

if __name__ == "__main__":
    main()