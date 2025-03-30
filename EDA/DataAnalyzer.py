
class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def calculate_mean_pace(self):
        return self.df['average pace (min/km)'].mean()

    def calculate_correlations(self):
        return self.df[['distance (m)', 'elapsed time (min)', 'elevation gain (m)', 'average heart rate (bpm)',
                        'average pace (min/km)']].corr()


