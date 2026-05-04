import math
import random

import numpy as np
import torch
import torch.nn.functional as F


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


def getNeighbors(i, width):
    x = i % width
    y = math.floor(i / width)
    up = (y - 1) * width + x
    down = (y + 1) * width + x
    if x == 0:
        return [-1, up, up + 1, -1, i + 1, -1, down, down + 1]
    elif x == width - 1:
        return [up - 1, up, -1, i - 1, -1, down - 1, down, -1]
    else:
        return [up - 1, up, up + 1, i - 1, i + 1, down - 1, down, down + 1]


def getGrid(i, mineCount, hidden, width):
    x = i % width
    y = math.floor(i / width)
    gridCords = [
        [x - 2, y - 2],
        [x - 1, y - 2],
        [x, y - 2],
        [x + 1, y - 2],
        [x + 2, y - 2],
        [x - 2, y - 1],
        [x - 1, y - 1],
        [x, y - 1],
        [x + 1, y - 1],
        [x + 2, y - 1],
        [x - 2, y],
        [x - 1, y],
        [x, y],
        [x + 1, y],
        [x + 2, y],
        [x - 2, y + 1],
        [x - 1, y + 1],
        [x, y + 1],
        [x + 1, y + 1],
        [x + 2, y + 1],
        [x - 2, y + 2],
        [x - 1, y + 2],
        [x, y + 2],
        [x + 1, y + 2],
        [x + 2, y + 2],
    ]
    grid = []
    for j in range(25):
        i = gridCords[j][1] * width + gridCords[j][0]
        if (
            gridCords[j][0] < 0
            or gridCords[j][0] >= width
            or gridCords[j][1] < 0
            or gridCords[j][1] >= width
        ):  # Out of bounds
            grid.append(-2)
        elif hidden[i] == 1:  # Hidden cell
            grid.append(-1)
        else:
            grid.append(mineCount[i])
    return grid


def getGlobalDensity(mines, hidden):
    m = 0
    for cell in mines:
        m += cell
    h = 0
    for cell in hidden:
        h += cell
    if h == 0:
        return 0
    return m / h


def setupGame(width):
    mines = []
    mineCount = []
    hidden = []
    for i in range(width**2):
        # 1 is mine
        if random.random() <= 0.2:
            mines.append(1)
        else:
            mines.append(0)
        # 1 is hidden
        hidden.append(1)

        mineCount.append(0)
    for i in range(width**2):
        neighbors = getNeighbors(i, width)
        for neighbor in neighbors:
            if neighbor > -1 and neighbor < width**2:
                mineCount[i] += mines[neighbor]
    return mines, mineCount, hidden


def predictionToColor(p):
    g = [0, 57, 0]
    r = [57, 0, 0]
    return [int(r[0] + p * (g[0] - r[0])), int(r[1] + p * (g[1] - r[1]))]


def printGame(mines, mineCount, hidden, width, predictions, bestPrediction):
    output = []
    numColors = [
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
        [0, 0, 140],
        [140, 0, 0],
        [0, 122, 124],
        [124, 0, 124],
        [75, 75, 75],
    ]
    for i in range(width**2):
        if hidden[i] == 1:  # Hidden cell
            hiddenColor = [198, 198, 198]
            if i in predictions:
                c = predictionToColor(predictions[i])
                hiddenColor = [198 + c[0], 198 + c[1], 198]
            text = "  "
            if i == bestPrediction:
                text = "GS"
            output.append(colorOutput(text, [120, 120, 0], hiddenColor))
        elif mines[i] == 1:  # Mine cell
            output.append(colorOutput("BM", [0, 0, 0], [255, 0, 0]))
        else:
            if mineCount[i] == 0:  # Cell with no number
                output.append(colorOutput("  ", [0, 0, 0], [180, 180, 180]))
            else:  # Cell with number
                output.append(
                    colorOutput(
                        f"{mineCount[i]} ", numColors[mineCount[i] - 1], [180, 180, 180]
                    )
                )
    printListAsBox(output, width)


def updateHidden(i, mines, mineCount, hidden, width):
    hidden[i] = 0
    neighbors = getNeighbors(i, width)
    for n in neighbors:
        if n > -1 and n < width**2:
            if mines[n] == 0:
                if hidden[n] == 1:
                    hidden[n] = 0
                    if mineCount[n] == 0:
                        hidden = updateHidden(n, mines, mineCount, hidden, width)
    return hidden


# Initialize model
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")


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


# predict_cell(model, getGrid(i, mineCount, hidden, width), globalDensity, device)
model = MinesweeperModel().load(
    torch.load("models/model-1363793rows-200steps.pth", map_location=device), device
)

won = False
while not won:
    # Initialize game
    gameSize = random.randint(5, 30)
    print(f"Starting new {gameSize}x{gameSize} game.")
    mines, mineCount, hidden = setupGame(gameSize)
    # Choose random cell to select that isnt a mine
    r = -1
    while r == -1 or mines[r] == 1:
        r = random.randint(0, gameSize**2 - 1)
    hidden = updateHidden(r, mines, mineCount, hidden, gameSize)
    # Game loop
    guesses = 0

    while True:
        predictions = {-1: 0.0}
        bestPrediction = -1
        gDensity = getGlobalDensity(mines, hidden)
        if int(gDensity) == 1:
            print("\nGame Won!")
            printGame(mines, mineCount, hidden, gameSize, predictions, -1)
            won = True
            break
        for i in range(gameSize**2):
            grid = getGrid(i, mineCount, hidden, gameSize)
            validGrid = False
            for g in grid:
                if g >= 0:
                    validGrid = True
                    break
            if hidden[i] == 1 and validGrid:
                p = predictCell(model, grid, gDensity, device)
                predictions[i] = p
                if p > predictions[bestPrediction]:
                    bestPrediction = i
        guesses += 1
        print(f"Current game state (guess {guesses}):")
        print(
            f"Selecting Cell ({bestPrediction % gameSize}, {bestPrediction // gameSize}) with a prediction score of {predictions[bestPrediction]}"
        )
        printGame(mines, mineCount, hidden, gameSize, predictions, bestPrediction)
        hidden = updateHidden(bestPrediction, mines, mineCount, hidden, gameSize)
        if mines[bestPrediction] == 1:
            print("Game Over!")
            print(f"The accuracy for this game: {(guesses - 1) / guesses * 100}%")
            printGame(mines, mineCount, hidden, gameSize, predictions, -1)
            break
# add detection for when game is won
