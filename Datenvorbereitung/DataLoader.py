import pandas as pd


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path, sep=";", engine="python")

    def clean_data(self):
        try:
            self.df.drop(columns=["Unnamed: 8", "Unnamed: 9", "time in min"], errors="ignore", inplace=True)
            numeric_cols = ["distance (m)", "elevation gain (m)", "average heart rate (bpm)", "elapsed time (s)"]
            self.df[numeric_cols] = self.df[numeric_cols].apply(pd.to_numeric, errors="coerce")

            self.df.dropna(subset=["gender"], inplace=True)

            self.df['elapsed time (min)'] = self.df['elapsed time (s)'] / 60
            self.df['average pace (min/km)'] = self.df['elapsed time (min)'] / (self.df['distance (m)'] / 1000)

            self.df = self.df[self.df['average pace (min/km)'] <= 15]
            self.df = self.df[self.df['average pace (min/km)'] >= 3]
            self.df = self.df[self.df['distance (m)'] <= 60000]
            self.df = self.df[self.df['elapsed time (min)'] <= 350]
            self.df = self.df[self.df['elapsed time (min)'] >= 5]
            self.df = self.df[self.df["elevation gain (m)"] <= 2000]
            self.df = self.df[(self.df['average heart rate (bpm)'] <= 250)]
            self.df = self.df[(self.df['average heart rate (bpm)'] >= 60)]

            return self.df
        except Exception as e:
            print(f"Fehler bei der Bereinigung der Daten: {e}")
            self.df = None

    def marathon_data(self):

        if self.df is None or self.df.empty:
            print("Fehler: DataFrame ist leer oder None.")
            return None

        try:
            marathon_df = self.df[(self.df['distance (m)'] >= 41650) & (self.df['distance (m)'] <= 42550)].copy()

            if marathon_df.empty:
                print("Keine Marathonläufe gefunden.")
                return None


            marathon_df['pace'] = marathon_df['elapsed time (min)'] / marathon_df['distance (m)']

            marathon_df['elapsed time (min)'] = marathon_df['elapsed time (min)'] + (42200 - marathon_df['distance (m)']) * \
                                           marathon_df['pace']
            marathon_df['distance (m)'] = 42200

            result_df = marathon_df[
                ['athlete', 'timestamp', 'distance (m)', 'elapsed time (min)']].copy()
            result_df['num_runs'] = result_df.groupby('athlete')['athlete'].transform('count')

            return result_df.drop_duplicates(subset='athlete', keep='first')

        except Exception as e:
            print(f"Fehler beim Extrahieren der Marathon-Daten: {e}")
            return None

    def half_marathon_data(self):

        if self.df is None or self.df.empty:
            print("Fehler: DataFrame ist leer oder None.")
            return None

        try:
            half_marathon_df = self.df[(self.df['distance (m)'] >= 21000) & (self.df['distance (m)'] <= 21200)].copy()

            if half_marathon_df.empty:
                print("Keine Halbmarathonläufe gefunden.")
                return None

            half_marathon_df['pace'] = half_marathon_df['elapsed time (min)'] / half_marathon_df['distance (m)']

            half_marathon_df['elapsed time (min)'] = half_marathon_df['elapsed time (min)'] + (
                        21100 - half_marathon_df['distance (m)']) * \
                                                     half_marathon_df['pace']
            half_marathon_df['distance (m)'] = 21100

            result_df = half_marathon_df[
                ['athlete', 'timestamp', 'distance (m)', 'elapsed time (min)']].copy()
            result_df['num_runs'] = result_df.groupby('athlete')['athlete'].transform('count')

            return result_df.drop_duplicates(subset='athlete', keep='first')

        except Exception as e:
            print(f"Fehler beim Extrahieren der Halbmarathon-Daten: {e}")
            return None
