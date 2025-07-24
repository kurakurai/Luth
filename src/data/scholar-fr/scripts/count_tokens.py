import pandas as pd
import tiktoken  


df = pd.read_csv(r"")  #dataset csv

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


text_columns = [""]  #dataset columns

def count_tokens(row):
    total = 0
    for col in text_columns:
        if pd.notna(row.get(col)):
            total += len(encoding.encode(str(row[col])))
    return total


df['token_count'] = df.apply(count_tokens, axis=1)
total_tokens = df['token_count'].sum()

print(f"Token count : {total_tokens}")
print(f"mean per sample : {df['token_count'].mean():.2f}")