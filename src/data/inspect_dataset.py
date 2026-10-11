
import pyarrow.parquet as pq
from pathlib import Path

path = Path(
    "data/raw/shopping_queries_dataset_products.parquet"
)

parquet_file = pq.ParquetFile(path)

print("File:", path.name)
print(f"Rows: {parquet_file.metadata.num_rows:,}")
print(f"Columns: {parquet_file.metadata.num_columns}")
print(f"Row groups: {parquet_file.metadata.num_row_groups}")

print("\nSchema:")
print(parquet_file.schema)

print("\nFirst 5 rows:")
preview = parquet_file.read_row_group(0).slice(0, 5)
print(preview.to_pandas().to_string(index=False))
