import pandas as pd
import os
import math
from tqdm import tqdm

def split_large_csv():
    # 配置
    source_dir = "data/datasets/wdc_lspm"
    table_id = "wdc_lspm"
    input_csv = os.path.join(source_dir, "tables", f"{table_id}.csv")
    output_dir = os.path.join(source_dir, "tables", "chunks")
    
    # 每个物理文件的行数 (建议 10000 行，这样文件不大，IO 友好)
    CHUNK_ROWS = 10000 
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Splitting {input_csv} into chunks of {CHUNK_ROWS} rows...")
    
    # 估算总行数用于进度条
    # total_rows = 11425521
    # num_chunks = math.ceil(total_rows / CHUNK_ROWS)
    
    chunk_idx = 0
    # 使用 chunksize 迭代读取，内存友好
    reader = pd.read_csv(input_csv, chunksize=CHUNK_ROWS)
    
    for df_chunk in tqdm(reader, desc="Splitting"):
        # 生成文件名: wdc_lspm_part_0.csv
        out_name = f"{table_id}_part_{chunk_idx}.csv"
        out_path = os.path.join(output_dir, out_name)
        
        # 保存 chunk
        df_chunk.to_csv(out_path, index=False)
        chunk_idx += 1
        
    print(f"✅ Split complete. Created {chunk_idx} files in {output_dir}")

if __name__ == "__main__":
    split_large_csv()
