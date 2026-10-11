
from pathlib import Path
import pyarrow.parquet as pq

path = Path(
    "data/raw/shopping_queries_dataset_examples.parquet"
)

parquet_file = pq.ParquetFile(path)

print("File:", path.name)
print(f"Rows: {parquet_file.metadata.num_rows:,}")
print("Columns:", parquet_file.schema_arrow.names)
print("\nSchema:")
print(parquet_file.schema_arrow)

print("\nFirst 10 rows:")
for batch in parquet_file.iter_batches(batch_size=10):
    print(batch.to_pandas().to_string(index=False))
    break
