import csv


def esc_tex(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")


if __name__ == "__main__":
    with open("./output/data/pareto_table.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = []

        for r in reader:
            model = esc_tex(r["Model"])
            dataset = esc_tex(r["Dataset"])
            lang = esc_tex(r["Language"])
            gain = float(r["Matched-Task (Cross-Language) gain"])
            off = float(r["Mean off-task"])
            rows.append(f"{model} & {dataset} & {lang} & {gain:.2f} & {off:.2f} \\\\")

    for line in rows:
        print(line)
