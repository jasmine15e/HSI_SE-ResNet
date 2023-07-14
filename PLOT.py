import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

train_df = pd.read_csv('')

plt.figure(dpi=100)

sns.set_style('whitegrid')
g = sns.lmplot(x='Salvianolic', y='Salvianolic_pred', data=train_df, hue='label')
sns.lmplot(x='Dihydrotanshinone', y='Dihydrotanshinone_pred', data=train_df, hue='label')
sns.lmplot(x='Cryptotanshinone', y='Cryptotanshinone_pred', data=train_df, hue='label')
sns.lmplot(x='Tanshinone', y='Tanshinone_pred', data=train_df, hue='label')
sns.lmplot(x='Moisture', y='Moisture_pred', data=train_df, hue='label')
g.fig.set_size_inches(10, 8)
plt.show()