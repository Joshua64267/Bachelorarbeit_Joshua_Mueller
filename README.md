# README

## Projektbeschreibung
Dieses Projekt befasst sich mit der Vorhersage von Laufzeiten bei Halbmarathons und Marathons anhand verschiedener Machine-Learning-Modelle. Es verwendet Daten aus einer Kaggle-Datenbank und bietet verschiedene Methoden zur Datenanalyse, Modellierung und Evaluierung.

## Voraussetzungen
### Installierte Bibliotheken
Stelle sicher, dass die folgenden Python-Bibliotheken installiert sind:

```bash
pip install numpy pandas lightgbm scipy shap matplotlib scikit-learn mlxtend fonttools pmdarima statsmodels
```

Falls zusätzliche Module benötigt werden, installiere sie mit `pip`.

## Nutzung
Das Hauptskript `main.py` kann mit verschiedenen Argumenten ausgeführt werden:

```bash
python main.py [OPTIONEN]
```

### Mögliche Argumente:
| Argument                 | Beschreibung |
|--------------------------|-------------|
| `--shap`                 | Inkludiert die SHAP-Funktion zur Modellinterpretation. |
| `--hyperparameter`       | Führt eine Hyperparameter-Optimierung durch. |
| `--eda`                  | Führt eine explorative Datenanalyse (EDA) durch. |
| `--ohne-hf`              | Verwendet ein Feature-Set ohne Herzfrequenz-Daten. |
| `--marathon-datensatz`   | Nutzt den Marathon-Datensatz statt des Halbmarathon-Datensatzes. |

### Beispiel:
Führe das Skript mit SHAP-Analyse und Hyperparameter-Tuning aus:
```bash
python main.py --shap --hyperparameter
```

## Struktur des Codes
### Hauptkomponenten:
- **Datenverarbeitung** (`Datenvorbereitung`):
  - `DataLoader`: Lädt und bereinigt die Rohdaten.
  - `Features`: Extrahiert relevante Features aus den Daten.
  - `FeaturesOhneHF`: Variante ohne Herzfrequenz-Features.
- **Explorative Datenanalyse** (`EDA`):
  - `DataVisualizer`: Erstellt verschiedene Plots zur Datenanalyse.
  - `DataAnalyzer`: Berechnet Korrelationen zwischen Variablen.
- **Machine Learning Modelle** (`Algorithms`):
  - `LGBMModel`: LightGBM-Modell.
  - `DecisionTreeModel`: Entscheidungsbaum-Modell.
  - `RandomForest`: Random-Forest-Modell.
  - `ARIMA`: Zeitreihenmodell für Vorhersagen.
  - `PowerLaws`: Modellierung mittels Potenzgesetzen.
- **Evaluation** (`Evaluation`):
  - Methoden zur Bewertung der Modelle (MAE, R², MSE, RMSE).

## Ergebnisse & Evaluation
Die verschiedenen Modelle werden mit den Metriken `Mean Absolute Error (MAE)`, `Mean Squared Error (MSE)`, `Root Mean Squared Error (RMSE)` und `R² Score` bewertet. Falls `--shap` angegeben wird, erfolgt zusätzlich eine Feature-Interpretation mittels SHAP-Werten.


## Autor
Joshua Müller

