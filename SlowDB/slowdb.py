import sqlite3
from contextlib import contextmanager
from os import PathLike
from pathlib import Path

import boto3


class SlowDB:
    def __init__(
        self,
        s3_bucket: str,
        s3_key: str,
        local_db_path: PathLike = "local_slowdb.db",
        readonly: bool = False,
        debug: bool = False,
    ):
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.local_db_path = Path(local_db_path)
        self.s3_client = boto3.client("s3")
        self.conn = None
        self.readonly = readonly
        self.debug = debug

    def download_db(self):
        self.s3_client.download_file(
            self.s3_bucket, self.s3_key, str(self.local_db_path)
        )

    def upload_db(self):
        # Create the bucket if it doesn't exist
        try:
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
        except self.s3_client.exceptions.ClientError:
            self.s3_client.create_bucket(Bucket=self.s3_bucket)

        self.s3_client.upload_file(str(self.local_db_path), self.s3_bucket, self.s3_key)

    def __enter__(self):
        self.download_db()
        self.conn = sqlite3.connect(self.local_db_path)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.readonly:
            # If the database is read-only, we don't want to commit any changes
            self.conn.close()
            return

        # If the database is not read-only, we'll commit the changes and upload the database
        self.conn.commit()
        self.conn.close()
        self.upload_db()
        if self.local_db_path.exists() and not self.debug:
            self.local_db_path.unlink()


@contextmanager
def connect(*args, **kwargs):
    db = SlowDB(*args, **kwargs)
    try:
        yield db.__enter__()
    except Exception as e:
        # If an exception occurs, we want to make sure the database is stored locally
        # so we can debug it
        db.debug = True

        # Re-raise the exception so the caller can handle
        raise e
    finally:
        db.__exit__(None, None, None)


# Example usage
if __name__ == "__main__":
    S3_BUCKET = "pl-prediction"
    S3_KEY = "2024/data.db"

    with connect(S3_BUCKET, S3_KEY) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM premier_league_managers WHERE club LIKE 'ARSENAL%'"
        )
        print(cursor.fetchall())
