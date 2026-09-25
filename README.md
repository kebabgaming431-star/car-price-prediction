# Used Car Price Prediction

Proiect de regresie care estimează prețul în USD al unei mașini second-hand. Soluția acoperă fluxul complet: analiză exploratorie, curățare, ingineria caracteristicilor, preprocesare, compararea modelelor, evaluare și salvarea modelului final.

## Setul de date

Fișierul `data/cars.csv` conține 56.244 de anunțuri și 12 coloane. Variabila țintă este `priceUSD`. Predictori: marcă, model, an, stare, kilometraj, combustibil, capacitatea motorului, culoare, transmisie, tracțiune și segment.

Problemele identificate includ 87 de duplicate, 47 de valori lipsă la capacitatea motorului, 1.905 la tracțiune și 5.291 la segment, plus câteva valori fizic improbabile pentru kilometraj și cilindree.

## Structura proiectului

```text
car-price-prediction/
├── data/
│   └── cars.csv
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
├── models/
│   └── car_price_model.joblib
├── reports/
│   ├── model_comparison.csv
│   ├── prediction_examples.csv
│   └── training_metadata.json
├── predict_price.py
├── requirements.txt
└── README.md
```

## Curățare și caracteristici noi

- denumirile `mileage(kilometers)` și `volume(cm3)` sunt simplificate;
- valorile categorice sunt normalizate cu litere mici și spații eliminate;
- duplicatele și rândurile imposibile sunt eliminate;
- valorile lipsă numerice sunt imputate cu mediana, iar cele categorice cu moda;
- `car_age` exprimă vechimea față de anul 2020, potrivit perioadei din set;
- `mileage_per_year` exprimă intensitatea utilizării;
- `log_mileage` reduce asimetria kilometrajului;
- `engine_volume_liters` oferă cilindreea într-o unitate mai ușor de interpretat;
- `make_model` surprinde combinația marcă-model.

Preprocesarea folosește `StandardScaler` pentru variabile numerice și `OneHotEncoder` pentru cele categorice. Ținta este transformată cu `log1p` în timpul antrenării și readusă automat în USD la predicție.

## Modele comparate

Toate modelele folosesc aceeași împărțire train/test: 80%/20%, `random_state=42`.

- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor

Comparația folosește MAE, MSE, RMSE și R². Modelul final este ales după cel mai mic MAE, deoarece această metrică arată direct eroarea medie în USD. Valorile obținute în rularea inclusă se găsesc în `reports/model_comparison.csv` și `reports/training_metadata.json`.

## Rezultate obținute

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| Extra Trees | 1.075,01 | 2.675,47 | 0,9029 |
| Random Forest | 1.116,09 | 2.968,98 | 0,8804 |
| Ridge | 1.294,60 | 3.762,09 | 0,8080 |
| Decision Tree | 1.313,59 | 2.975,37 | 0,8799 |

Modelul final este **Extra Trees Regressor**. El are cel mai mic MAE și explică aproximativ 90,3% din variația prețului pe setul de test. MAE de aproximativ 1.075 USD înseamnă că predicția diferă în medie cu circa 1.075 USD față de prețul real.

## Instalare și rulare

Necesită Python 3.10 sau mai nou.

```bash
python -m venv .venv
```

Activare Windows:

```bash
.venv\Scripts\activate
```

Activare macOS/Linux:

```bash
source .venv/bin/activate
```

Instalare biblioteci:

```bash
pip install -r requirements.txt
```

Antrenarea și compararea tuturor modelelor:

```bash
python -m src.model_training --data data/cars.csv --output-dir .
```

Evaluarea modelului salvat:

```bash
python -m src.model_evaluation --data data/cars.csv --model models/car_price_model.joblib
```

Predicție pentru exemplul Volkswagen Golf din cerință:

```bash
python predict_price.py
```

Exemplu personalizat:

```bash
python predict_price.py --make bmw --car-model 3-seriya --year 2012 --mileage 150000 --fuel-type diesel --volume 2000 --transmission auto --drive-unit "rear drive" --segment D
```

## Notebook

Pornește Jupyter din rădăcina proiectului:

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

Notebook-ul prezintă EDA, deciziile de curățare, caracteristicile create, rezultatele modelelor și exemple de predicții.

## Reproductibilitate

Modelul salvat este un pipeline complet. Include imputarea, scalarea, codificarea, regresorul și transformarea țintei. Poate primi date brute cu aceleași câmpuri fără preprocesare manuală suplimentară, după aplicarea funcțiilor de curățare și inginerie incluse în proiect.
