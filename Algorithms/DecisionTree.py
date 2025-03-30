import shap
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


class DecisionTreeModel:
    def __init__(self, criterion="squared_error", max_depth=None, random_state=42):
        self.model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, random_state=random_state)

    def train(self, X_train, y_train, hyperparameter_tuning=False):

        if hyperparameter_tuning:
            param_grid = {
                "criterion": ["squared_error", "friedman_mse"],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }

            grid_search = GridSearchCV(
                DecisionTreeRegressor(random_state=42),
                param_grid,
                cv=3,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            print(f"Beste Parameter: {grid_search.best_params_}")
            self.model = grid_search.best_estimator_

        else:
            self.model = DecisionTreeRegressor(criterion='squared_error', min_samples_split=10, min_samples_leaf=1,
                                               random_state=42, max_depth=None)
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    @staticmethod
    def evaluate(y_true, y_pred):
        print("\nVorhersagen vs. Tatsächliche Zeiten:")
        for actual, predicted in zip(y_true, y_pred):
            print(f"Tatsächliche Zeit: {actual:.2f} Minuten - Vorhergesagte Zeit: {predicted:.2f} Minuten")

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        print("")
        print(f"Mean Absolute Error (MAE): {mae:.2f} min")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.2f} Minuten")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Minuten")

        return mae, r2, mse, rmse
    def shap(self, X_train, X_test):

        explainer = shap.Explainer(lambda x: self.model.predict(x), X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test)
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        plt.figure(figsize=(4, 4))
        plt.subplots_adjust(left=0.4)
        plt.suptitle("Decision Tree")
        shap.waterfall_plot(shap_values[0], max_display=len(X_train.columns))

    def get_model(self):
        return self.model
