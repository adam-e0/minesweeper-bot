import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

import login

load_dotenv()
if not all(
    [os.getenv("DB_USERNAME"), os.getenv("DB_PASSWORD"), os.getenv("DB_SCHEMA")]
):
    print("Error: DB_USERNAME, DB_PASSWORD, or DB_SCHEMA environment variable not set.")
    exit(1)


def createMinesweeperDataset(limit):
    # Load the SQL data and return the grid features, density, and targets
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
        query = f"SELECT * FROM {schema}.minesweeper_dataset;"
        if limit > 0:
            query = f"SELECT * FROM {schema}.minesweeper_dataset LIMIT {limit};"
        with db.cursor() as c:
            c.execute(query)
            rows = c.fetchall()
            column_names = [desc[0] for desc in c.description]

        # Build numpy arrays from the fetched rows with explicit type conversion
        grid_features_list = []
        global_density_list = []
        targets_list = []
        for row in rows:
            row_dict = dict(zip(column_names, row))
            grid_values = []
            for col in column_names:
                if col not in ("safe", "index", "global_density"):
                    raw_value = row_dict[col]
                    grid_values.append(int(raw_value))
            grid_features_list.append(grid_values)
            global_density_list.append(float(row_dict["global_density"]))
            targets_list.append(int(row_dict["safe"]))

        gridFeatures = np.array(grid_features_list, dtype=np.int32)
        globalDensity = np.array(global_density_list, dtype=np.float32)
        targets = np.array(targets_list, dtype=np.int32)
        return gridFeatures, globalDensity, targets
    except BaseException as e:
        print(f"Error loading dataset from database: {e}")
        raise


def addModelToDB(model_name, steps_trained, rows_trained, trained_accuracy):
    schema = os.getenv("DB_SCHEMA")
    success, error = login.login(
        os.getenv("DB_USERNAME"), os.getenv("DB_PASSWORD"), schema
    )
    if not success:
        print(f"Login failed: {error}")
        exit(1)
    db = login.db()
    if db is None:
        print("ERROR: Database connection is None!")
        return
    try:
        # First, check if the model already exists
        checkQuery = f"SELECT 1 FROM {schema}.models WHERE model_name = %s;"

        with db.cursor() as c:
            c.execute(checkQuery, (model_name,))
            result = c.fetchone()
            exists = result is not None

        if exists:
            # Update existing row
            updateQuery = f"UPDATE {schema}.models SET steps_trained = %s, rows_trained = %s, trained_accuracy = %s WHERE model_name = %s;"
            with db.cursor() as c:
                c.execute(
                    updateQuery,
                    (steps_trained, rows_trained, trained_accuracy, model_name),
                )
            print(f"Updated {model_name} in models database table")
        else:
            # Insert new row
            insertQuery = f"INSERT INTO {schema}.models (model_name, steps_trained, rows_trained, trained_accuracy) VALUES (%s, %s, %s, %s);"
            with db.cursor() as c:
                c.execute(
                    insertQuery,
                    (model_name, steps_trained, rows_trained, trained_accuracy),
                )
            print(f"Added {model_name} to models database table")

        db.commit()
    except BaseException as e:
        if db is not None:
            db.rollback()
        print(f"Error adding model to database: {e}")


def trainModels():
    try:
        limit = int(
            input(
                "Enter the max number of rows of data to train the model(s) on (0 will use all rows): "
            )
        )
        if limit < 0:
            raise ValueError
    except ValueError:
        print("Please enter a valid positive integer.")
        return

    try:
        numIntervals = int(
            input(
                "Enter the number of intervals to divide the dataset into (1 will train the full dataset on 1 model): "
            )
        )
        if numIntervals <= 0:
            raise ValueError
    except ValueError:
        print("Please enter a valid positive integer.")
        return

    try:
        epochs = int(input("Enter the number of training steps per model: "))
        if epochs <= 0:
            raise ValueError
    except ValueError:
        print("Please enter a valid positive integer.")
        return

    print("Loading full dataset from SQL table")
    gridFeatures, globalDensity, targets = createMinesweeperDataset(limit)
    totalRows = len(gridFeatures)

    os.makedirs("models", exist_ok=True)

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    print(f"Total dataset size: {totalRows} rows. Training on {device}.\n")

    for i in range(1, numIntervals + 1):
        currentRowCount = int(totalRows * (i / numIntervals))

        print(f"Model {i}/{numIntervals} training on {currentRowCount} rows of data:")

        batchSize = 128
        subsetIndices = list(range(currentRowCount))

        params = {
            "conv1_w": nn.Parameter(torch.randn(32, 11, 3, 3, device=device)),
            "conv1_b": nn.Parameter(torch.randn(32, device=device)),
            "conv2_w": nn.Parameter(torch.randn(64, 32, 3, 3, device=device)),
            "conv2_b": nn.Parameter(torch.randn(64, device=device)),
            "fc1_w": nn.Parameter(torch.randn(32, 65, device=device)),
            "fc1_b": nn.Parameter(torch.randn(32, device=device)),
            "fc2_w": nn.Parameter(torch.randn(1, 32, device=device)),
            "fc2_b": nn.Parameter(torch.randn(1, device=device)),
        }
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(list(params.values()), lr=0.001)
        epochAcc = 0.0
        for i in range(epochs):
            totalLoss = 0
            correctPreds = 0
            totalSamples = 0

            np.random.shuffle(subsetIndices)

            numBatches = (currentRowCount + batchSize - 1) // batchSize

            for batchIdx in range(numBatches):
                batchStart = batchIdx * batchSize
                batchEnd = min(batchStart + batchSize, currentRowCount)
                batchIndices = subsetIndices[batchStart:batchEnd]

                batchData = []

                for idx in batchIndices:
                    # Get a single sample from the dataset
                    cells = gridFeatures[idx]
                    cells = np.insert(cells, 12, -1)
                    cellsShifted = torch.tensor(cells + 2, dtype=torch.long)
                    gridOnehot = F.one_hot(cellsShifted, num_classes=11).float()
                    grid2d = gridOnehot.view(5, 5, 11).permute(2, 0, 1)
                    density = torch.tensor([globalDensity[idx]], dtype=torch.float32)
                    target = torch.tensor([targets[idx]], dtype=torch.float32)
                    batchData.append([grid2d, density, target])

                gridBatchData = []
                densityBatchData = []
                targetBatchData = []
                for d in batchData:
                    gridBatchData.append(d[0])
                    densityBatchData.append(d[1])
                    targetBatchData.append(d[2])

                gridBatch = torch.stack(gridBatchData).to(device)
                densityBatch = torch.stack(densityBatchData).to(device)
                targetBatch = torch.stack(targetBatchData).to(device)

                optimizer.zero_grad()

                # Forward pass through the network
                conv1_w, conv1_b = params["conv1_w"], params["conv1_b"]
                conv2_w, conv2_b = params["conv2_w"], params["conv2_b"]
                fc1_w, fc1_b = params["fc1_w"], params["fc1_b"]
                fc2_w, fc2_b = params["fc2_w"], params["fc2_b"]

                predictions = F.conv2d(gridBatch, conv1_w, conv1_b)
                predictions = F.relu(predictions)
                predictions = F.conv2d(predictions, conv2_w, conv2_b)
                predictions = F.relu(predictions)
                predictions = predictions.view(predictions.size(0), -1)
                predictions = torch.cat((predictions, densityBatch), dim=1)
                predictions = F.relu(F.linear(predictions, fc1_w, fc1_b))
                predictions = F.linear(predictions, fc2_w, fc2_b)

                loss = criterion(predictions, targetBatch)
                loss.backward()
                optimizer.step()

                totalLoss += loss.item()
                predsBinary = (torch.sigmoid(predictions) >= 0.5).float()
                correctPreds += (predsBinary == targetBatch).sum().item()
                totalSamples += targetBatch.size(0)

            epochLoss = totalLoss / numBatches
            epochAcc = (correctPreds / totalSamples) * 100

            print(
                f"\nEpoch {i + 1}/{epochs}:\nLoss: {epochLoss}\nAccuracy: {epochAcc}%"
            )

        modelName = f"model-{currentRowCount}rows-{i + 1}steps.pth"
        torch.save(params, "./models/" + modelName)
        print(f"\nSaved model: {modelName}")
        addModelToDB(modelName, epochs, currentRowCount, epochAcc)


trainModels()
