import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def convert_file(args):
    csv_path, parquet_path = args
    try:
        # 读取 CSV
        df = pd.read_csv(csv_path)
        # 保存为 Parquet
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        return True
    except Exception as e:
        print(f"Error converting {csv_path}: {e}")
        return False

def main():
    chunks_dir = "data/datasets/wdc_lspm/tables/chunks"
    if not os.path.exists(chunks_dir):
        print(f"Dir not found: {chunks_dir}")
        return

    files = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".csv")])
    print(f"Found {len(files)} CSV chunks. Converting to Parquet...")
    
    tasks = []
    for f in files:
        csv_path = os.path.join(chunks_dir, f)
        parquet_path = os.path.join(chunks_dir, f.replace(".csv", ".parquet"))
        if not os.path.exists(parquet_path):
            tasks.append((csv_path, parquet_path))
    
    if not tasks:
        print("All files already converted.")
        return

    # 并行转换 (为了稳妥，用 16 个 worker，避免再把 IO 打爆)
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(convert_file, tasks), total=len(tasks), desc="Converting"))
        
    print(f"✅ Converted {sum(results)} files.")

if __name__ == "__main__":
    main()
