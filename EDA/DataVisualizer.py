import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def plot_univariate(self):
        plt.figure(figsize=(14, 10))

        # Distanz
        plt.subplot(3, 2, 1)
        sns.histplot(self.df['distance (m)'], kde=True, color='skyblue')
        plt.title('Verteilung der Distanz')

        # Elapsed Time (in Minuten)
        plt.subplot(3, 2, 2)
        sns.histplot(self.df['elapsed time (min)'], kde=True, color='orange')
        plt.title('Verteilung der verstrichenen Zeit (in Minuten)')

        # Elevation Gain
        plt.subplot(3, 2, 3)
        sns.histplot(self.df['elevation gain (m)'], kde=True, color='green')
        plt.title('Verteilung der Höhenmeter')

        # Durchschnittliche Herzfrequenz
        plt.subplot(3, 2, 4)
        sns.histplot(self.df['average heart rate (bpm)'], kde=True, color='red')
        plt.title('Verteilung der durchschnittlichen Herzfrequenz')

        # Durchschnittliche Pace
        plt.subplot(3, 2, 5)
        sns.histplot(self.df['average pace (min/km)'], kde=True, color='purple')
        plt.title('Verteilung der durchschnittlichen Pace')

        plt.tight_layout()
        plt.show()

    def plot_boxplots(self):
        plt.figure(figsize=(14, 10))

        # Distanz
        plt.subplot(3, 2, 1)
        sns.boxplot(y=self.df['distance (m)'], color='skyblue')
        plt.title('Boxplot der Distanz')

        # Elapsed Time (in Minuten)
        plt.subplot(3, 2, 2)
        sns.boxplot(y=self.df['elapsed time (min)'], color='orange')
        plt.title('Boxplot der verstrichenen Zeit (in Minuten)')

        # Elevation Gain
        plt.subplot(3, 2, 3)
        sns.boxplot(y=self.df['elevation gain (m)'], color='green')
        plt.title('Boxplot der Höhenmeter')

        # Durchschnittliche Herzfrequenz
        plt.subplot(3, 2, 4)
        sns.boxplot(y=self.df['average heart rate (bpm)'], color='red')
        plt.title('Boxplot der durchschnittlichen Herzfrequenz')

        # Durchschnittliche Pace
        plt.subplot(3, 2, 5)
        sns.boxplot(y=self.df['average pace (min/km)'], color='purple')
        plt.title('Boxplot der durchschnittlichen Pace')

        plt.tight_layout()
        plt.show()

    def plot_scatter(self):
        cmap = sns.color_palette("coolwarm_r", as_cmap=True)
        norm = plt.Normalize(self.df['average pace (min/km)'].min(), self.df['average pace (min/km)'].max())

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.df['distance (m)'], self.df['elapsed time (min)'],
                              c=self.df['average pace (min/km)'], cmap=cmap, norm=norm, s=100, edgecolors='white',
                              linewidth=1)

        cbar = plt.colorbar(scatter)
        cbar.set_label("Pace (min/km)")

        plt.title('Verstrichene Zeit vs. Distanz ')
        plt.ylabel('Verstrichene Zeit (min)')
        plt.xlabel('Distanz (m)')
        plt.show()

    def plot_heatmap(self):
        plt.figure(figsize=(16, 10))
        correlation_matrix = self.df[
            ['distance (m)', 'elapsed time (min)', 'elevation gain (m)', 'average heart rate (bpm)',
             'average pace (min/km)']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Korrelation zwischen Variablen')
        plt.show()


    def plot_pace_analysis(self):
        cmap = sns.color_palette("coolwarm_r", as_cmap=True)
        norm = plt.Normalize(self.df['average pace (min/km)'].min(), self.df['average pace (min/km)'].max())

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.df['distance (m)'], self.df['average heart rate (bpm)'],
                              c=self.df['average pace (min/km)'], cmap=cmap, norm=norm, s=100, edgecolors='white',
                              linewidth=1)

        cbar = plt.colorbar(scatter)
        cbar.set_label("Pace (min/km)")


        plt.title('Distanz vs. Herzfrequenz (bpm)')
        plt.ylabel('Herzfrequenz (bpm)')
        plt.xlabel('Distanz (m)')
        plt.show()

    def heartrate_test(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)

        slowest_20 = self.df.nlargest(20, 'average pace (min/km)')
        print(slowest_20)

        avg_hr = slowest_20['average heart rate (bpm)'].mean()
        print(f"Durchschnittliche Herzfrequenz der langsamsten 20: {avg_hr:.2f} bpm")

    def plot_scatter_heartrate(self):
        cmap = sns.color_palette("coolwarm_r", as_cmap=True)
        norm = plt.Normalize(self.df['average heart rate (bpm)'].min(), self.df['average heart rate (bpm)'].max())

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.df['distance (m)'], self.df['elapsed time (min)'],
                              c=self.df['average heart rate (bpm)'], cmap=cmap, norm=norm, s=100, edgecolors='white',
                              linewidth=1)

        cbar = plt.colorbar(scatter)
        cbar.set_label("Herzfrequenz (bpm)")

        plt.title('Verstrichene Zeit vs. Distanz')
        plt.ylabel('Verstrichene Zeit (min)')
        plt.xlabel('Distanz (m)')
        plt.show()

