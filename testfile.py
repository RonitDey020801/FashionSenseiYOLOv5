import pandas as pd
df = pd.read_csv("styles.csv", engine='python', on_bad_lines='skip')
print(df.columns)