import pandas as pd
import matplotlib.pyplot as plt

class TitanicEDA:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
    
    def get_summary_stats(self):
        return self.df.describe()
    
    def plot_survival_by_pclass(self):
        plt.figure(figsize=(8, 6))
        self.df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
        plt.title('Survival Rate by Passenger Class')
        plt.xlabel('Passenger Class')
        plt.ylabel('Survival Rate')
        plt.savefig('survival_by_pclass.png')
        
    def plot_survival_by_sex(self):
        plt.figure(figsize=(8, 6))
        self.df.groupby('Sex')['Survived'].mean().plot(kind='bar')
        plt.title('Survival Rate by Sex')
        plt.xlabel('Sex')
        plt.ylabel('Survival Rate')
        plt.savefig('survival_by_sex.png')
        
    def plot_survival_by_age(self):
        plt.figure(figsize=(8, 6))
        self.df.hist('Age', by=self.df['Survived'], bins=20)
        plt.title('Survival Rate by Age')
        plt.xlabel('Age')
        plt.ylabel('Survival Rate')
        plt.savefig('survival_by_age.png')
        
# Example usage
titanic_eda = TitanicEDA('C:/Users/omkar/OneDrive/Desktop/MLOPS assignments/Assignment1/train.csv')
print(titanic_eda.get_summary_stats())
titanic_eda.plot_survival_by_pclass()
titanic_eda.plot_survival_by_sex()
titanic_eda.plot_survival_by_age()