import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

class RandomForest:
    def __init__(self):
        self.model = None

    def prepare_data(self, features):
        X = features.drop(columns=['athlete', 'elapsed_time (m)'])
        y = features['elapsed_time (m)']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train, hyperparameter_tuning=False):

        if hyperparameter_tuning:
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 7, 10, 20, None],
                "min_samples_split": [2, 5, 10, 15],
                "min_samples_leaf": [1, 2, 4, 7],
                "random_state": [31, 42, 54, 60]

            }

            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=3,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            print(f"Beste Parameter: {grid_search.best_params_}")
            return grid_search.best_estimator_

        else:
            self.model = RandomForestRegressor(n_estimators=100, min_samples_split= 10, min_samples_leaf= 1, random_state=42, max_depth=7)
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        # Ausgabe für die ersten paar Vorhersagen
        for true, pred in zip(y_test, y_pred):
            print(f"Tatsächliche Zeit: {true  :.2f} Minuten - Vorhergesagte Zeit: {pred :.2f} Minuten")
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        print(f"R²-Score: {r2_score(y_test, y_pred):.4f}")
        print(f"MAE: {mean_absolute_error(y_test, y_pred)  :.2f} Minuten")
        print(f"Mean Squared Error (MSE): {mse :.2f} Minuten")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Minuten")

    def shap(self, X_train, X_test):
        explainer = shap.Explainer(lambda x: self.model.predict(x), X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        plt.figure(figsize=(4, 4))
        plt.subplots_adjust(left=0.4)
        plt.suptitle("Random Forest")
        shap.waterfall_plot(shap_values[0], max_display=len(X_train.columns))

