import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class FeaturesOhneHF:
    @staticmethod
    def get_runners_data(athlete_id, timestamp, elapsed_time, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], format="%d.%m.%y")
        timestamp = pd.to_datetime(timestamp, format="%d.%m.%y")

        athlete_df = df[(df['athlete'] == athlete_id) & (df['timestamp'] < timestamp)]

        if athlete_df.empty:
            return pd.DataFrame()

        total_distance = athlete_df["distance (m)"].sum()
        total_time_min = athlete_df["elapsed time (s)"].sum() / 60
        #avg_heart_rate = athlete_df["average heart rate (bpm)"].mean()

        monthly_avg_pace, pace_change_percentage, a_pace, b_pace = FeaturesOhneHF.calculate_pace_change(athlete_df)
        a_power, b_power = FeaturesOhneHF.calculate_power_law(athlete_df)
        #time_percentage_in_zone_One,time_percentage_in_zones_Two , time_percentage_in_zones_Three, time_percentage_in_zones_Four, time_percentage_in_zones_Five = FeatursOhneHF.categorize_heart_rate(athlete_df)

        data = {
            "athlete": [athlete_id],
            "gender": [athlete_df["gender"].iloc[0]],
            "elapsed time (min)": elapsed_time,
            "total_elevation_gain": [athlete_df["elevation gain (m)"].sum()],
            "total_training_time": [total_time_min],
            #"average heart rate (bpm)": [avg_heart_rate],
            #"max_heart_rate": [athlete_df["average heart rate (bpm)"].max()],
            #"min_heart_rate": [athlete_df["average heart rate (bpm)"].min()],
            #"std_heart_rate": [athlete_df["average heart rate (bpm)"].std()],
            #"median_heart_rate": [athlete_df["average heart rate (bpm)"].median()],
            "average_run_distance": [athlete_df["distance (m)"].mean()],
            "total_runs": [len(athlete_df)],
            "max_distance": [athlete_df["distance (m)"].max()],
            "min_distance": [athlete_df["distance (m)"].min()],
            "std_a_distance": [athlete_df["distance (m)"].std()],
            "median_distance": [athlete_df["distance (m)"].median()],
            "total_distance": [total_distance],
            "average_pace": [(total_time_min / (total_distance / 1000)) if total_distance > 0 else None],
            "max_pace": [((athlete_df["elapsed time (s)"] / 60 )/ (athlete_df["distance (m)"] / 1000)).max()],
            "min_pace": [((athlete_df["elapsed time (s)"] / 60)/ (athlete_df["distance (m)"] / 1000)).min()],
            "median_pace": [((athlete_df["elapsed time (s)"] / 60)/ (athlete_df["distance (m)"] / 1000)).median()],
            "std_a_pace": [((athlete_df["elapsed time (s)"] / 60) / (athlete_df["distance (m)"] / 1000)).std()],
            "pace_change_percentage": [pace_change_percentage],
            "pace_model_a": [a_pace],
            "pace_model_b": [b_pace],
            "power_law_a": [a_power],
            "power_law_b": [b_power],
            #"time_percentage_in_zone_One": [time_percentage_in_zone_One],
            #"time_percentage_in_zone_Two": [time_percentage_in_zones_Two],
            #"time_percentage_in_zone_Three": [time_percentage_in_zones_Three],
            #"time_percentage_in_zone_Four": [time_percentage_in_zones_Four],
            #"time_percentage_in_zone_Five": [time_percentage_in_zones_Five],
        }


        return pd.DataFrame(data)

    @staticmethod
    def calculate_pace_change(athlete_df):
        monthly_avg_pace = athlete_df.groupby(athlete_df['timestamp'].dt.to_period('M'))['elapsed time (s)'].mean() / (
                athlete_df.groupby(athlete_df['timestamp'].dt.to_period('M'))['distance (m)'].mean() / 1000)

        initial_pace = monthly_avg_pace.iloc[0]
        final_pace = monthly_avg_pace.iloc[-1]
        pace_change_percentage = (final_pace - initial_pace) / initial_pace * 100
        months = np.array(range(len(monthly_avg_pace))).reshape(-1, 1)
        pace_values = monthly_avg_pace.values
        reg = LinearRegression().fit(months, pace_values)
        a = reg.intercept_
        b = reg.coef_[0]

        return monthly_avg_pace, pace_change_percentage, a, b

    @staticmethod
    def calculate_power_law(athlete_df):
        distances = athlete_df['distance (m)']
        log_distances = np.log(distances[distances > 0])
        log_counts = np.log(np.arange(1, len(log_distances) + 1))

        reg = LinearRegression().fit(log_counts.reshape(-1, 1), log_distances)
        a = np.exp(reg.intercept_)
        b = reg.coef_[0]

        return a, b

    @staticmethod
    def get_running_data_for_all_athletes(marathon_data, df):
        all_athletes_data = []

        for athlete_id in marathon_data['athlete']:

            athlete_data = FeaturesOhneHF.get_runners_data(
                athlete_id,
                marathon_data[marathon_data['athlete'] == athlete_id]['timestamp'].max(),
                marathon_data[marathon_data['athlete'] == athlete_id]['elapsed time (min)'].values[0],
                df
            )

            if not athlete_data.empty:
                athlete_data['gender'] = athlete_data['gender'].map({'M': 0, 'F': 1})
                all_athletes_data.append(athlete_data)

        if all_athletes_data:
            return pd.concat(all_athletes_data, ignore_index=True)
        else:
            return pd.DataFrame()

    @staticmethod
    def prepare_data(features):

        X = features.drop(columns=['athlete', 'elapsed time (min)'])
        y = features['elapsed time (min)']
        return train_test_split(X, y, test_size=0.2, random_state=42)
