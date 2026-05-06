import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

import login

load_dotenv()
if not all(
    [os.getenv("DB_USERNAME"), os.getenv("DB_PASSWORD"), os.getenv("DB_SCHEMA")]
):
    print("Error: DB_USERNAME, DB_PASSWORD, or DB_SCHEMA environment variable not set.")
    exit(1)


class MinesweeperModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_w = torch.nn.Parameter(None)
        self.conv1_b = torch.nn.Parameter(None)
        self.conv2_w = torch.nn.Parameter(None)
        self.conv2_b = torch.nn.Parameter(None)
        self.fc1_w = torch.nn.Parameter(None)
        self.fc1_b = torch.nn.Parameter(None)
        self.fc2_w = torch.nn.Parameter(None)
        self.fc2_b = torch.nn.Parameter(None)

    def load(self, params, device):
        for key, param in params.items():
            setattr(self, key, torch.nn.Parameter(param.to(device)))
        return self

    def forward(self, grid, density):
        predictions = F.conv2d(grid, self.conv1_w, self.conv1_b)
        predictions = F.relu(predictions)
        predictions = F.conv2d(predictions, self.conv2_w, self.conv2_b)
        predictions = F.relu(predictions)
        predictions = predictions.view(predictions.size(0), -1)
        predictions = torch.cat((predictions, density), dim=1)
        predictions = F.relu(F.linear(predictions, self.fc1_w, self.fc1_b))
        predictions = F.linear(predictions, self.fc2_w, self.fc2_b)
        return predictions


def colorOutput(input, textRGB, highRGB):
    textCode = f"\033[38;2;{textRGB[0]};{textRGB[1]};{textRGB[2]}m"
    highCode = f"\033[48;2;{highRGB[0]};{highRGB[1]};{highRGB[2]}m"
    return f"{textCode}{highCode}{input}\033[0m"


def printListAsBox(list, width):
    for i in range(len(list)):
        if i % width == 0 and i != 0:
            print()
        print(list[i], end="")
    print()


def printHeatMap(grid, safeValue):
    minimum = min(grid)
    maximum = max(grid)
    heatMap = []
    for i in range(len(grid)):
        g = grid[i]
        color = [0, 0, 0]
        if g >= 0:
            color[1] = int(g / maximum * 255)
        else:
            color[0] = 255 - int(g / minimum * 255)
        heatMap.append(colorOutput("  ", [0, 0, 0], color))
        if i == 11:
            safeColor = [0, 0, 0]
            if safeValue >= 0.5:
                safeColor[1] = int((safeValue - 0.5) / 0.5 * 255)
            else:
                safeColor[0] = 255 - int(safeValue / 0.5 * 255)
            heatMap.append(colorOutput("  ", [0, 0, 0], safeColor))
    printListAsBox(heatMap, 5)


def fetchMinesweeperDataset(limit):
    schema = os.getenv("DB_SCHEMA")
    success, error = login.login(
        os.getenv("DB_USERNAME"), os.getenv("DB_PASSWORD"), schema
    )
    if not success:
        print(f"Login failed: {error}")
        exit(1)
    db = login.db()
    if db is None:
        raise Exception("Database connection is None!")
    try:
        terminator = ";"
        if limit > 0:
            terminator = f" LIMIT {limit};"  # f" TABLESAMPLE ({limit} ROWS);"
        # query = f"SELECT * FROM {schema}.minesweeper_dataset{terminator}"
        query = f"""
        SELECT * FROM (
            SELECT * FROM {schema}.minesweeper_dataset TABLESAMPLE bernoulli(100) WHERE safe = 0
            UNION ALL
            SELECT * FROM {schema}.minesweeper_dataset TABLESAMPLE bernoulli(100) WHERE safe = 1
        ) combined
        ORDER BY random(){terminator}"""
        with db.cursor() as c:
            c.execute(query)
            rows = c.fetchall()
            columnNames = [desc[0] for desc in c.description]
        table = {"grid": [], "safe": [], "global_density": []}
        for row in rows:
            rowDict = dict(zip(columnNames, row))
            gridValues = []
            for col in columnNames:
                if col not in ("safe", "index", "global_density"):
                    gridValues.append(int(rowDict[col]))
                if col == "safe":
                    table["safe"].append(int(rowDict[col]))
                elif col == "global_density":
                    table["global_density"].append(float(rowDict[col]))
            table["grid"].append(gridValues)
        return table
    except BaseException as e:
        print(f"Error loading dataset from database: {e}")
        raise


def fetchDatasetAverages(limit):
    schema = os.getenv("DB_SCHEMA")
    success, error = login.login(
        os.getenv("DB_USERNAME"), os.getenv("DB_PASSWORD"), schema
    )
    if not success:
        print(f"Login failed: {error}")
        exit(1)
    db = login.db()
    if db is None:
        raise Exception("Database connection is None!")
    try:
        limitClause = ""
        if limit > 0:
            limitClause = f" LIMIT {limit}"
        cells = [
            f"cell_{dy}_{dx}"
            for dy in range(-2, 3)
            for dx in range(-2, 3)
            if not (dx == 0 and dy == 0)
        ]
        avg_selects = ",\n            ".join(
            f'AVG(CASE WHEN "{col}" = -2 THEN 0.0 WHEN "{col}" = -1 THEN -0.25 ELSE "{col}" / 4.0 END) AS "avg_{col}"'
            for col in cells
        )
        query = f"""
        SELECT
            safe,
            {avg_selects}
        FROM (
            SELECT * FROM (
                SELECT * FROM {schema}.minesweeper_dataset TABLESAMPLE bernoulli(100) WHERE safe = 0
                UNION ALL
                SELECT * FROM {schema}.minesweeper_dataset TABLESAMPLE bernoulli(100) WHERE safe = 1
            ) combined
            ORDER BY random(){limitClause}
        ) sampled_data
        GROUP BY
            safe
        ORDER BY
            safe DESC;
        """
        with db.cursor() as c:
            c.execute(query)
            rows = c.fetchall()
            columnNames = [desc[0] for desc in c.description]
        grids = [[], []]
        for row in rows:
            rowDict = dict(zip(columnNames, row))
            safeVal = 0
            for col in columnNames:
                if col == "safe":
                    safeVal = int(rowDict[col])
                else:
                    grids[safeVal].append(float(rowDict[col]))
        # safe, notsafe
        return grids[1], grids[0]
    except BaseException as e:
        print(f"Error loading dataset from database: {e}")
        raise


def predictCell(model, grid, globalDensity, device):
    # 1. Put the model in evaluation mode (disables training behaviors like Dropout)
    model.eval()
    # 2. Format the Grid
    # Convert input to a flat numpy array (handles both 1D and 2D lists automatically)
    gridFlat = np.array(grid).flatten()
    if len(gridFlat) != 25:
        raise ValueError(
            f"Expected exactly 25 cells for a 5x5 grid, got {len(gridFlat)}"
        )
    # Shift values so there are no negative numbers: [-2 to 8] becomes [0 to 10]
    cells_shifted = torch.tensor(gridFlat + 2, dtype=torch.long)
    # One-hot encode the 11 possible states
    grid_onehot = F.one_hot(cells_shifted, num_classes=11).float()
    # Reshape to (Channels, Height, Width) -> (11, 5, 5)
    grid_2d = grid_onehot.view(5, 5, 11).permute(2, 0, 1)
    # ADD BATCH DIMENSION: (11, 5, 5) -> (1, 11, 5, 5)
    grid_batch = grid_2d.unsqueeze(0).to(device)
    # 3. Format the Density
    # Create a tensor of shape (1, 1)
    density_batch = torch.tensor([[globalDensity]], dtype=torch.float32).to(device)
    # 4. Make the Prediction
    # torch.no_grad() tells PyTorch not to calculate gradients, saving RAM and speeding it up
    with torch.no_grad():
        output = model(grid_batch, density_batch)
        # Convert raw output to a probability between 0 and 1 using Sigmoid
        probability = torch.sigmoid(output).item()
    return probability


def normalize(num):
    if num == -2:
        return 0
    if num == -1:
        return -1 / 4
    if num > 0:
        return num / 4
    return num


def avgGrids(grids, safeValues):
    safeGrid = []
    notSafeGrid = []
    for i in range(24):
        safeGrid.append(0)
        notSafeGrid.append(0)
    safeCount = 0
    notSafeCount = 0

    for i in range(len(safeValues)):
        if safeValues[i] >= 0.5:
            safeCount += 1
            for j in range(len(safeGrid)):
                safeGrid[j] += normalize(grids[i][j])
        elif safeValues[i] < 0.5:
            notSafeCount += 1
            for j in range(len(notSafeGrid)):
                notSafeGrid[j] += normalize(grids[i][j])

    for i in range(len(safeGrid)):
        safeGrid[i] = safeGrid[i] / safeCount

    for i in range(len(notSafeGrid)):
        notSafeGrid[i] = notSafeGrid[i] / notSafeCount

    return safeGrid, notSafeGrid


try:
    limit = int(input("Enter the rows of data analize (0 will use all rows): "))
    if limit < 0:
        raise ValueError
except ValueError:
    print("Please enter a valid positive integer.")
    exit(-1)

models = os.listdir("models/")
print("All models:")
for model in models:
    if model.split(".").pop() != "pth":
        models.pop(models.index(model))
for i in range(len(models)):
    print(f"{i}: {models[i]}")
modelsToBench = {}
try:
    modelsInput = input(
        "Select which models to benchmark using a comma seperated list of their indicies (-1 will select them all): "
    ).strip()
    if len(modelsInput) == 0:
        raise ValueError
    if "," in modelsInput:
        for model in modelsInput.split(","):
            m = int(model.strip())
            if m < 0 or m >= len(models):
                raise ValueError
            modelsToBench[m] = {}
            modelsToBench[m]["safeGrid"] = []
            modelsToBench[m]["notSafeGrid"] = []
        print(modelsToBench)
    elif int(modelsInput) == -1:
        for i in range(len(models)):
            modelsToBench[i] = {}
            modelsToBench[i]["safeGrid"] = []
            modelsToBench[i]["notSafeGrid"] = []
    else:
        m = int(modelsInput.strip())
        if m < 0 or m >= len(models):
            raise ValueError
        modelsToBench[m] = {}
        modelsToBench[m]["safeGrid"] = []
        modelsToBench[m]["notSafeGrid"] = []
except ValueError:
    print("Please enter a valid input.")
    exit(-1)

print("Loading in data from table...")
table = fetchMinesweeperDataset(limit)
print("Done loading data")

# Initialize device
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

for m in modelsToBench.keys():
    # Initialize model
    model = MinesweeperModel().load(
        torch.load(f"models/{models[m]}", map_location=device), device
    )
    print(f"Now analizing model: {models[m]}")

    modelPrediction = []

    for i in range(len(table["safe"])):
        grid = list(table["grid"][i])
        grid.insert(12, -1)
        modelPrediction.append(
            predictCell(model, grid, table["global_density"][i], device)
        )
        if (i + 1) % (len(table["safe"]) // 10) == 0:
            print(
                f"Model prediction progress: {(i + 1) / len(table['safe']) * 100}% ({i + 1}/{len(table['safe'])})"
            )

    modelsToBench[m]["safeGrid"], modelsToBench[m]["notSafeGrid"] = avgGrids(
        table["grid"], modelPrediction
    )

safeGrid, notSafeGrid = fetchDatasetAverages(limit)

print()
print("Dataset safe grids:")
printHeatMap(safeGrid, 1)
print("Dataset not safe Grids: ")
printHeatMap(notSafeGrid, 0)

for m in modelsToBench.keys():
    print()
    print(models[m])
    print("Predicted safe grids:")
    printHeatMap(modelsToBench[m]["safeGrid"], 1)
    print("Predicted not Safe Grids: ")
    printHeatMap(modelsToBench[m]["notSafeGrid"], 0)
