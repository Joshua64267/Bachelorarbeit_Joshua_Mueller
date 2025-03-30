import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


class LGBMModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, hyperparameter_tuning=False):
        if hyperparameter_tuning:
            param_grid = {
                'learning_rate': [0.01, 0.03, 0.1, 0.15, 0.2],
                'max_depth': [3, 5, 9, 13, 15],
                'n_estimators': [50, 100, 200, 500, 700],
                'num_leaves': [10, 20, 30, 60, 70],
                'min_data_in_leaf': [1, 5, 10, 30, 40],
            }
            grid_search = GridSearchCV(
                lgb.LGBMRegressor(), param_grid, cv=5,
                scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print("Beste Parameter:", grid_search.best_params_)
            print("Bestes MAE:", -grid_search.best_score_)

        else:
            self.model = lgb.LGBMRegressor(learning_rate=0.01, n_estimators=100, max_depth=3,
                                           num_leaves=10, min_data_in_leaf=10, verbose= -1)
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        print("\nVorhersagen vs. Tatsächliche Zeiten:")
        for actual, predicted in zip(y_test, y_pred):
            print(f"Tatsächliche Zeit: {actual:.2f} Minuten - Vorhergesagte Zeit: {predicted:.2f} Minuten")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        print(f"\nMean Absolute Error (MAE): {mae:.2f} Minuten")
        print(f"Mean Squared Error (MSE): {mse:.2f} Minuten")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Minuten")
        print(f"R²-Score: {r2:.4f}")

    def shap(self, X_train, X_test):
        explainer = shap.Explainer(self.model, X_train)
        shap_values = explainer(X_test, check_additivity=False)
        shap.summary_plot(shap_values, X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        plt.figure(figsize=(4, 4))
        plt.subplots_adjust(left=0.4)
        plt.suptitle("LGBM")
        shap.waterfall_plot(shap_values[0], max_display=len(X_train.columns))