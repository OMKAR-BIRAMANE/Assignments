# titanic_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TitanicEDA:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)

    def summary_statistics(self):
        return self.data.describe()

    def plot_survival_rate(self, feature):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature, y='Survived', data=self.data)
        plt.title(f'Survival Rate by {feature}')
        plt.savefig(f'survival_rate_by_{feature}.png')
        plt.close()

    def plot_age_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='Age', hue='Survived', kde=True, bins=30)
        plt.title('Age Distribution by Survival')
        plt.savefig('age_distribution.png')
        plt.close()
