import argparse
import pandas as pd
from Algorithms.LGBM import LGBMModel
from Datenvorbereitung.DataLoader import DataLoader
from Datenvorbereitung.Features import Features
from Datenvorbereitung.FeaturesOhneHF import FeaturesOhneHF
from EDA.DataVisualizer import DataVisualizer
from Algorithms.ARIMA import ARIMAModel
from Algorithms.PowerLaws import PowerLaws
from Algorithms.RandomForest import RandomForest
from Algorithms.DecisionTree import DecisionTreeModel
from EDA.DataAnalyzer import DataAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shap', action= "store_true", help='Inkludiert die SHAP Funktion')
    parser.add_argument('--hyperparameter', action="store_true", help='Inkludiert die Hyperparameter Funktion')
    parser.add_argument('--eda', action="store_true", help='Inkludiert die EDA Funktionen')
    parser.add_argument('--ohne-hf', action="store_true", help='Nimmt die Features ohne Herzfrequenz')
    parser.add_argument('--marathon-datensatz', action="store_true", help='Nimmt den Marathon Datensatz')
    args = parser.parse_args()


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path = "./Datensatz/raw-data-kaggle.csv"
    loader = DataLoader(file_path)
    loader.load_data()
    df = loader.clean_data()
    marathon_times = loader.marathon_data()
    halb_marathon_times = loader.half_marathon_data()
    datensatz = halb_marathon_times
    distance = 21100



    if args.marathon_datensatz:
        datensatz = marathon_times
        distance = 42200

    if args.ohne_hf:
        running_data = FeaturesOhneHF.get_running_data_for_all_athletes(datensatz, df)
        X_train, X_test, y_train, y_test = FeaturesOhneHF.prepare_data(running_data)
    else:
        running_data = Features.get_running_data_for_all_athletes(datensatz, df)
        X_train, X_test, y_train, y_test = Features.prepare_data(running_data)


    if args.eda:
        data_analyzer = DataAnalyzer(df)
        visualizer = DataVisualizer(df)


        correlation_matrix = data_analyzer.calculate_correlations()
        print("Korrelationen zwischen den Variablen:")
        print(correlation_matrix)

        visualizer.plot_univariate()
        visualizer.plot_boxplots()
        visualizer.plot_scatter()
        visualizer.plot_heatmap()
        visualizer.plot_pace_analysis()
        visualizer.heartrate_test()
        visualizer.plot_scatter_heartrate()

    "--------------------------------------------------------------------"

    "Decision Tree"
    print("Decision Tree")
    decision_tree = DecisionTreeModel()
    decision_tree.train(X_train, y_train, hyperparameter_tuning= args.hyperparameter)
    predictions = decision_tree.predict(X_test)

    #Evaluation
    DecisionTreeModel.evaluate(y_test, predictions)

    #SHAP
    if args.shap:
        decision_tree.shap(X_train, X_test)

    "-------------------------------------------------------------"
    "LGBM"
    print("\nLGBM\n")
    lgbm = LGBMModel()
    lgbm.train(X_train, y_train, args.hyperparameter)
    y_pred = lgbm.predict(X_test)

    #Evaluation
    lgbm.evaluate(y_test, y_pred)

    #SHAP
    if args.shap:
        lgbm.shap(X_train, X_test)

    "-------------------------------------------------------------"
    "Random Forest"

    print("\nRandom Forest:\n")
    random_forest = RandomForest()
    random_forest.train_model(X_train, y_train, hyperparameter_tuning= args.hyperparameter)

    #Evaluation
    random_forest.evaluate(X_test, y_test)

    #SHAP
    if args.shap:
        random_forest.shap(X_train, X_test)

    "-------------------------------------------------------------"
    "Power Low"
    print("\nPower Low:\n")
    power_laws = PowerLaws()
    power_laws.run_predictions(datensatz, distance, df)

    # Evaluation
    power_laws.evaluate()

    "ARIMA"
    print("\nARIMA:\n")
    model = ARIMAModel(df)
    model.get_all_predictions(datensatz, distance)


if __name__ == "__main__":
    main()