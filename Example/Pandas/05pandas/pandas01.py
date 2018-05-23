import pandas as pd
import numpy as np

df = pd.DataFrame({'key1':list('aabba'),
                  'key2': ['one','two','one','two','one'],
                  'data1': np.random.randn(5),
                  'data2': np.random.randn(5)})

print(df)

grouped = df['data1'].groupby(df['key1'])
print(grouped)

print('1111222222-----')

print(grouped.mean())

print(df.groupby('key1').mean())

