import os

import pandas as pd

directory = '/Users/mrfishpl/Desktop/GradKNN/searches_results'

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        max_row = df.loc[df['last_task_accuracy'].idxmax()]
        print(f"File: {filename}")
        print(max_row)
        print("\n")
