# https://inloop.github.io/sqlite-viewer/
import sqlite3
import os


def get_schema_info(db_path: str) -> str:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        schema_info = []

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for (table_name,) in tables:
            # Get columns for this table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            table_info = f"Table: {table_name}\n"
            table_info += "\n".join(f"  - {col[1]} ({col[2]})" for col in columns)
            schema_info.append(table_info)

    return "\n\n".join(schema_info)


if __name__ == "__main__":
    src_dir = os.path.expanduser(
        "~/Documents/genai-training-pydanticai/data/text-to-sql"
    )
    db_file = "db/chinook.sqlite"
    print(get_schema_info(os.path.join(src_dir, db_file)))
