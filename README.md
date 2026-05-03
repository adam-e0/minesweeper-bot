# minesweeper-bot

Minesweeper AI bot

Dataset : https://www.kaggle.com/datasets/michelechierchia/dataset-minesweeper-game/data

Place `minesweeper_dataset.csv` in `/data/`

Create a .env file with the following variables:

```
DB_USERNAME=username
DB_PASSWORD=password
DB_HOST=host.name
DB_PORT=port#
DB_NAME=dbname
DB_SCHEMA=dbschema
```

Run `database-setup.py` to setup the database tables.

```
python3 database-setup.py
```

Run `train-model.py` to train model.

```
python3 train-model.py
```
