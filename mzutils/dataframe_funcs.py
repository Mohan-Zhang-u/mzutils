import pathlib
import pandas as pd


def parquet_append(filepath: pathlib.Path or str, df: pd.DataFrame) -> None:
    """
    Append to dataframe to existing .parquet file. Reads original .parquet file in, appends new dataframe, writes new .parquet file out.
    :param filepath: Filepath for parquet file.
    :param df: Pandas dataframe to append. Must be same schema as original.
    """
    import pyarrow
    table_original_file = pyarrow.parquet.read_table(source=filepath,  pre_buffer=False, use_threads=True, memory_map=True)  # Use memory map for speed.
    table_to_append = pyarrow.Table.from_pandas(df)
    table_to_append = table_to_append.cast(table_original_file.schema)  # Attempt to cast new schema to existing, e.g. datetime64[ns] to datetime64[us] (may throw otherwise).
    handle = pyarrow.parquet.ParquetWriter(filepath, table_original_file.schema)  # Overwrite old file with empty. WARNING: PRODUCTION LEVEL CODE SHOULD BE MORE ATOMIC: WRITE TO A TEMPORARY FILE, DELETE THE OLD, RENAME. THEN FAILURES WILL NOT LOSE DATA.
    handle.write_table(table_original_file)
    handle.write_table(table_to_append)
    handle.close()  # Writes binary footer. Until this occurs, .parquet file is not usable.
