import numpy as np
import pandas as pd
import scipy.optimize as opt
from matplotlib import pyplot as plt
from Evaluation.Evaluation import Evaluation


class PowerLaws:
    def __init__(self):
        self.df_results = None

    @staticmethod
    def power_law(x, a, b):
        return a * np.power(x, b)

    def get_runners_data(self, athlete_id, timestamp, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%d.%m.%y")
        timestamp = pd.to_datetime(timestamp, format="%d.%m.%y")

        one_year_before = timestamp - pd.DateOffset(years=1)

        athlete_df = df[
            (df['athlete'] == athlete_id) & (df['timestamp'] < timestamp) & (df['timestamp'] >= one_year_before)]

        return athlete_df

    def predict_times(self, athlete_id, timestamp,distanz, df):
        athlete_df = self.get_runners_data(athlete_id, timestamp, df)

        if athlete_df is None:
            return None

        x_data = athlete_df['distance (m)'].values
        y_data = athlete_df['elapsed time (s)'].values

        if np.any(y_data <= 0) or np.any(np.isnan(y_data)):
            print(f"Ungültige Daten für Läufer {athlete_id}!")
            return None

        try:
            params, _ = opt.curve_fit(self.power_law, x_data, y_data, bounds=(0, [np.inf, 2]))
            a, b = params
            predicted_time = self.power_law(distanz, *params)

            return predicted_time, a, b

        except Exception as e:
            print(f"Fehler bei der Berechnung: {e}")
            return None, None, None

    def visualize_fit(self, athlete_id, x_data, y_data, params):
        a, b = params
        plt.figure(figsize=(8, 5))
        plt.scatter(x_data, y_data, label="Echte Daten", color="blue")

        x_fit = np.linspace(min(x_data), max(x_data) * 1.2, 100)
        y_fit = self.power_law(x_fit, *params)
        plt.plot(x_fit, y_fit, label=f"Power-Law Fit: y = {a:.2f} * x^{b:.2f}", color="red")

        plt.xlabel("Distanz (m)")
        plt.ylabel("Zeit (s)")
        plt.title(f"Marathon-Zeit Vorhersage für Läufer {athlete_id}")
        plt.legend()
        plt.grid(True)
        plt.show()


    def run_predictions(self, marathon_data, distanz, df):
        data = marathon_data
        results = []
        for _, row in data.iterrows():
            athlete_id = row['athlete']
            timestamp = row['timestamp']
            actual_time = row['elapsed time (min)']
            predicted_time = self.predict_times(athlete_id, timestamp, distanz, df)

            results.append({
                'Läufer ID': athlete_id,
                'Tatsächliche Zeit (min)': actual_time,
                'Predicted Zeit (min)': predicted_time[0]/ 60,
                'a': predicted_time[1],
                'b': predicted_time[2]
            })
        self.df_results = pd.DataFrame(results)

    def evaluate(self):
        print("Power Law:")
        print("R²-Score:", Evaluation.r2(self.df_results))
        print("MAE:", Evaluation.mae(self.df_results))
        print(f"Mean Squared Error (MSE): {Evaluation.mse(self.df_results) :.2f} Minuten")
        print(f"Root Mean Squared Error (RMSE): {Evaluation.rmse(self.df_results):.2f} Minuten")
