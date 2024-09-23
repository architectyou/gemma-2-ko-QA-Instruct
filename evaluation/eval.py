from dotenv import load_dotenv

load_dotenv()

import pandas as pd

df = pd.read_csv("data/ragas_synthetic_dataset.csv")
df.head()