"""Manages an application-wide connection to the database."""

import os

# Change this to False and set the pyscopg version if you want to use
# pyscopg 2 or 3.
USE_PG8000 = True
PSYCOPG_VERSION = 3

if USE_PG8000:
    import pg8000 as connector
elif PSYCOPG_VERSION == 3:
    import psycopg as connector
elif PSYCOPG_VERSION == 2:
    import psycopg2 as connector
else:
    raise RuntimeError("Invalid database connection setting.")

DB = None


def login(username, password, schema):
    """
    Login and obtain a connection to the database, and set the search path.
    Return (True, None) on success, (False, error message) otherwise.
    """
    global DB
    if DB is not None:
        DB.close()
        DB = None

    host = os.getenv("DB_HOST")
    dbname = os.getenv("DB_NAME")
    port = os.getenv("DB_PORT")

    connect_info = {
        "user": username,
        "password": password,
        "host": host,
        "port": int(port),
    }
    if USE_PG8000:
        connect_info["database"] = dbname
    else:
        connect_info["dbname"] = dbname

    try:
        DB = connector.connect(**connect_info)
        del connect_info
        del password
    except BaseException as e:
        DB = None
        return False, str(e)

    stmt = "SELECT set_config('search_path', %s, false)"
    try:
        with DB.cursor() as cursor:
            cursor.execute(stmt, [schema])
        DB.commit()
    except BaseException as e:
        return False, str(e)

    return True, None


def db():
    """
    Return the previously opened database Connection object (or None). No
    checking is done to ensure the connection is open and valid.
    """
    return DB
