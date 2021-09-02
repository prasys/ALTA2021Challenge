import pandas as pd

train_df = pd.read_pickle('trainMinA.h5') #trainMinAS.h5
eval_df = pd.read_pickle('devMinA.h5') #devMinAS.h5


frames = [train_df, eval_df]
df = pd.concat(frames)


df = df.sample(frac=1).reset_index(drop=True)
df['labels'].replace({1: 'B', 0: 'A', 2: 'C'}, inplace=True)
df.drop_duplicates(inplace=True)
df.to_csv("googlecloud4.csv",index=False,header=None)