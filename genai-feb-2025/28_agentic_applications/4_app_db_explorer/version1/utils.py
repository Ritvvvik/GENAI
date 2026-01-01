import sqlite3
from config_reader import settings
import os


def get_chunks_by_field(db_file_path: str):
    with sqlite3.connect(db_file_path) as conn:
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

        chunks = []
        seq = 1
        for (table,) in tables:
            cols = cursor.execute(f"PRAGMA table_info({table})").fetchall()
            for col in cols:
                chunk = {
                    "id": seq,
                    "text": f"Table: {table[0]}, Column: {col[1]}, Type: {col[2]}",
                    "metadata": {"table": table[0], "column": col[1], "type": col[2]},
                }
                chunks.append(chunk)
                seq = seq + 1

        return chunks


def get_chunks_by_table(db_file_path: str):
    with sqlite3.connect(db_file_path) as conn:
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

        chunks = []
        seq = 1
        for (table,) in tables:
            cols = cursor.execute(f"PRAGMA table_info({table})").fetchall()
            text = []
            text.append(f"Table: {table}\n")
            for col in cols:
                text.append(f"  - {col[1]} ({col[2]})")
            text = "\n".join(text)
            chunk = {
                "id": seq,
                "text": text,
                "metadata": {"table": table},
            }
            chunks.append(chunk)
            seq = seq + 1

        return chunks


if __name__ == "__main__":
    db_file_path = os.path.join(
        settings.file_paths.src_dir, settings.file_paths.db_file
    )
    chunks = get_chunks_by_field(db_file_path)
    print(chunks)
    chunks = get_chunks_by_table(db_file_path)
    print(chunks)
