import re
import os
import pandas as pd
regex = re.compile('\S*.csv')
for root, dirs, files in os.walk('./'):
  for filename in files:
    if regex.match(filename):
      print(filename)
      df = pd.read_csv(filename)
      rowid = df['Value'].idxmax()
      print(df[rowid:rowid+1])

