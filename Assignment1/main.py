# main.py
from titanic_eda import TitanicEDA

def main():
    data_path = 'titanic.csv'  # Path to your Titanic dataset
    eda = TitanicEDA(data_path)
    eda.load_data()
    print(eda.summary_statistics())
    eda.plot_survival_rate('Pclass')
    eda.plot_survival_rate('Sex')
    eda.plot_age_distribution()

if __name__ == "__main__":
    main()
