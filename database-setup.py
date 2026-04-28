import csv
import os

from dotenv import load_dotenv

import login

# Load variables from .env file
load_dotenv()

# Use environment variables for database credentials
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
schema = os.getenv("DB_SCHEMA")

csvPath = "./data/minesweeper_dataset.csv"
tableName = f"{schema}.minesweeper_dataset"


def createDatasetTable():
    db = login.db()
    try:
        if not os.path.exists(csvPath):
            return f"Error: File {csvPath} not found.", None

        with open(csvPath, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames

            if not fieldnames:
                return "CSV file is empty", None

            columns = "index SERIAL PRIMARY KEY"
            for name in fieldnames:
                name = name.replace('"', "")
                dataType = "INT"
                if name == "global_density":
                    dataType = "FLOAT"
                columns += f', "{name}" {dataType} NOT NULL'

            createTableQuery = f"CREATE TABLE IF NOT EXISTS {tableName} ({columns});"

            print(createTableQuery)

            rows = []
            for row in reader:
                rows.append([row[name] for name in fieldnames])

        if db is None:
            return "Error: Database connection is not established.", None
        with db.cursor() as c:
            c.execute(createTableQuery)

        placeholders = ""
        for i in range(len(fieldnames)):
            if i > 0:
                placeholders += ", "
            placeholders += "%s"

        columns = ""
        for i in range(len(fieldnames)):
            if i > 0:
                columns += ", "
            name = fieldnames[i].replace('"', "")
            columns += f'"{name}"'

        insertQuery = f"INSERT INTO {tableName} ({columns}) VALUES ({placeholders});"

        print(insertQuery)

        count = 0
        if db is not None:
            with db.cursor() as c:
                for values in rows:
                    c.execute(insertQuery, values)
                    count += 1
                db.commit()
                return f"Successfully inserted {count} rows into '{tableName}'.", None
        else:
            return "Error: Database connection is not established.", None

    except BaseException as e:
        if db is not None:
            db.rollback()
        return None, str(e)


def createModelsTable():
    db = login.db()
    try:
        columns = (
            "model_steps_trained INT NOT NULL PRIMARY KEY, "
            "model_size INT NOT NULL, "
            "dataset_start_index INT, "
            "dataset_end_index INT"
        )

        createTableQuery = f"CREATE TABLE IF NOT EXISTS {schema}.models ({columns});"

        print(createTableQuery)

        if db is None:
            return "Error: Database connection is not established.", None
        with db.cursor() as c:
            c.execute(createTableQuery)
            db.commit()
            return f"Successfully created table '{schema}.models'.", None

    except BaseException as e:
        if db is not None:
            db.rollback()
        return None, str(e)


def createModelStatisticsTable():
    db = login.db()
    try:
        columns = (
            "model_steps_trained INT NOT NULL, "
            "games_played INT, "
            "games_won INT, "
            "total_guesses INT, "
            "correct_guesses INT, "
            "PRIMARY KEY (model_steps_trained), "
            "FOREIGN KEY (model_steps_trained) REFERENCES "
            f"{schema}.models(model_steps_trained)"
        )

        createTableQuery = (
            f"CREATE TABLE IF NOT EXISTS {schema}.model_statistics ({columns});"
        )

        print(createTableQuery)

        if db is None:
            return "Error: Database connection is not established.", None
        with db.cursor() as c:
            c.execute(createTableQuery)
            db.commit()
            return f"Successfully created table '{schema}.model_statistics'.", None

    except BaseException as e:
        if db is not None:
            db.rollback()
        return None, str(e)


if not all([username, password, schema]):
    print("Error: DB_USERNAME, DB_PASSWORD, or DB_SCHEMA environment variable not set.")
    exit(1)

success, error = login.login(username, password, schema)
if success:
    createDatasetTable()
    createModelsTable()
    createModelStatisticsTable()
else:
    print(f"Login failed: {error}")
