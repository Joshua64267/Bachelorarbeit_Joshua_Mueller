import warnings
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Datenvorbereitung.DataLoader import DataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)


class ARIMAModel:
    def __init__(self, df):
        self.df = df

    def prepare_data(self, athlete_id):
        df = self.df[self.df['athlete'] == athlete_id].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%y')
        df = df.sort_values('timestamp')
        df['pace'] = df['elapsed time (s)'] / df['distance (m)']
        return df.set_index('timestamp')['pace']

    def train_arima_model(self, time_series):
        model = auto_arima(time_series, seasonal=False, stepwise=True, suppress_warnings=True)
        return ARIMA(time_series, order=model.order).fit()

    def predict_time(self, model, distance):
        forecast_result = model.forecast()
        predicted_pace = forecast_result.iloc[0]
        predicted_time = predicted_pace * distance
        return predicted_time

    def evaluate_prediction(self, predicted_time, actual_time):
        error = abs(predicted_time - actual_time)
        print(f'Vorhergesagte Zeit: {predicted_time / 60:.2f} min')
        print(f'Reale Zeit: {actual_time:.2f} min')
        print(f'Fehler: {error / 60:.2f} min')

    def evaluate_predictions(self, predicted_times, actual_times):
        actual_times_sec = np.array(actual_times) * 60
        predicted_times_sec = np.array(predicted_times)

        mae = mean_absolute_error(actual_times_sec, predicted_times_sec)
        mse = mean_squared_error(actual_times_sec, predicted_times_sec)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_times_sec, predicted_times_sec)

        print(f"MAE (min): {mae / 60:.2f} min")
        print(f"MSE (h): {mse / 3600:.2f} h")
        print(f"RMSE (min): {rmse / 60:.2f} min")
        print(f"R² Score: {r2:.2f}")

    def get_all_predictions(self, dataset, distance):
        predicted_times = []
        actual_times = []

        for athlete_id in dataset['athlete'].unique():
            time_series = self.prepare_data(athlete_id)
            if len(time_series) < 10:
                continue

            model = self.train_arima_model(time_series)
            predicted_time = self.predict_time(model,distance)
            actual_time = dataset[dataset['athlete'] == athlete_id]['elapsed time (min)'].iloc[-1]
            print(f'Vorhergesagte Zeit: {predicted_time / 60:.2f} min')
            print(f'Reale Zeit: {actual_time:.2f} min')
            predicted_times.append(predicted_time)
            actual_times.append(actual_time)

        self.evaluate_predictions(predicted_times, actual_times)



def main():
    file_path = "../Datensatz/raw-data-kaggle.csv"
    loader = DataLoader(file_path)
    loader.load_data()
    df = loader.clean_data()
    marathon_times = loader.marathon_data()
    halb_marathon_times = loader.half_marathon_data()
    df = pd.DataFrame(df)
    print(marathon_times)
    model = ARIMAModel(df)

    model.get_all_predictions(halb_marathon_times,21100)



if __name__ == "__main__":
    main()