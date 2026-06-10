import pandas as pd
import json
import os

def build_kb():
    # 使用生成的 dataset 目录
    data_dir = "data/datasets/wdc_lspm"
    table_path = os.path.join(data_dir, "tables/wdc_lspm.csv")
    gt_path = os.path.join(data_dir, "gt.csv")
    output_path = os.path.join(data_dir, "wdc_products_kb.jsonl")

    if not os.path.exists(table_path) or not os.path.exists(gt_path):
        print(f"Missing files: {table_path} or {gt_path}")
        return

    print("Reading files...")
    # 只需要读取必要的列，避免内存爆炸
    try:
        df_table = pd.read_csv(table_path, usecols=['title'])
        df_gt = pd.read_csv(gt_path)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    print(f"Table rows: {len(df_table)}, GT rows: {len(df_gt)}")

    kb_map = {}

    # 遍历 GT，收集每个 Cluster ID 对应的第一个 Title 作为名称
    # 在真实场景中，可能需要更复杂的聚合，但这里作为 Baseline 足够
    for _, row in df_gt.iterrows():
        r_idx = int(row['row'])
        label = str(row['label']).strip()
        
        if r_idx < len(df_table):
            if label not in kb_map:
                desc = str(df_table.iloc[r_idx, 0]) # 假设 title 是第一列 (read_csv usecols 会重排吗？最好按名取)
                # usecols返回的列序可能不定，但在只有一列的情况下没问题
                # 更稳妥的方式:
                # desc = df_table.at[r_idx, 'title'] 
                # 但 iterrows 是慢的，且 df_table 索引必须是对齐的
                # 由于 prepare_wdc_lspm 保证了 table 和 gt 行对齐
                pass
    
    # 更高效的方法
    # 将 table 和 gt 合并
    df_merged = pd.concat([df_table, df_gt[['label']]], axis=1)
    
    # 按 label 分组，取第一个 title
    # 注意：wdc 数据量大，groupby 可能慢，但在 11M 数据上应该还行
    print("Grouping by label to find canonical names...")
    # dropna
    df_merged.dropna(subset=['label', 'title'], inplace=True)
    
    # 我们可以先去重，每个 label 只留一个
    df_unique = df_merged.drop_duplicates(subset=['label'], keep='first')
    
    print(f"Unique clusters found: {len(df_unique)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df_unique.iterrows():
            cluster_id = str(row['label'])
            name = str(row['title']).strip()
            
            entry = {
                "id": cluster_id,
                "name": name,
                "title": name,
                "text": name,
                "type": "ProductCluster",
                "path": "WDC > Product"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved KB to {output_path}")

if __name__ == "__main__":
    build_kb()
