import pandas as pd

def random_sample_tsv(input_file, output_file, sample_fraction=0.5, random_seed=42):
  """
  从 TSV 文件中随机抽取一定比例的行，并保存到新的 TSV 文件中。

  Args:
    input_file (str): 输入的 TSV 文件路径。
    output_file (str): 输出的 TSV 文件路径。
    sample_fraction (float, optional): 抽样的比例，默认为 0.5 (50%)。
    random_seed (int, optional): 随机种子，用于保证结果的可复现性。默认为 42。
  """
  try:
    # 使用 pandas 读取 TSV 文件。 [3, 10, 13, 17]
    # sep='\t' 指定了制表符为分隔符。 [13]
    df = pd.read_csv(input_file, sep='\t')

    # 使用 sample 方法进行随机抽样。 [5, 7, 11, 12]
    # frac 参数指定了抽样的比例。 [5, 7]
    # random_state 参数设置了随机种子，以确保每次运行结果相同。 [5, 7, 12, 15]
    sampled_df = df.sample(frac=sample_fraction, random_state=random_seed)

    # 将抽样后的数据框写入新的 TSV 文件。 [1, 4, 8, 16]
    # sep='\t' 指定了制表符为分隔符。 [1, 4]
    # index=False 表示不将索引写入文件。 [1]
    sampled_df.to_csv(output_file, sep='\t', index=False)

    print(f"成功从 {input_file} 中随机抽取 {sample_fraction*100}% 的数据并保存至 {output_file}")

  except FileNotFoundError:
    print(f"错误：找不到输入文件 {input_file}")
  except Exception as e:
    print(f"发生错误： {e}")


random_sample_tsv("/path/to/your/MMMU_DEV_VAL.tsv", "/path/to/saved/sample5.tsv", sample_fraction=0.5, random_seed=127)