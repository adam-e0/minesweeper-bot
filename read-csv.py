import csv

i = 0

with open("./data/minesweeper_dataset.csv", mode="r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        i += 1
        # print(row)  # Access data by column header
        for y_0 in range(5):
            for x_0 in range(5):
                # Convert 0 - 5 range to -2 - 2
                x = x_0 % 5 - 2
                y = 2 - (y_0 % 5)
                # print(f"({x}, {y}): ", end="")
                if x != 0 or y != 0:
                    cell = int(row[f"cell_{x}_{y}"])
                    if cell == -1:  # Hidden Cell
                        print("#", end=" ")
                    elif cell == -2:  # Out of Bounds
                        print(".", end=" ")
                    else:  # Center Cell
                        print(cell, end=" ")
                else:
                    if int(row["safe"]) == 1:
                        print("S", end=" ")
                    else:
                        print("B", end=" ")
            print()
        print()
        if i == 1000:
            break
