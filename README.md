# House Price Predictor — Gustavo A. Madero, Mexico City

Machine learning model that estimates the market price of a house for sale in the Gustavo A. Madero borough (Mexico City), built end to end following **CRISP-DM**.

> The notebook is written in Spanish.

## Problem

Mexico has no public record of individual home sale prices, so this project uses **public listings** from three real estate portals as asking prices. The model predicts the log of the **price per built m²**, and the final price is price/m² × built m².

## Data

- **2,152 listings** collected from Mercado Libre, Vivanuncios and Lamudi.
- No personal data is stored. The raw and cleaned datasets are **not included** in this repository to respect the portals' terms of use.

Main data quality issues found and handled:

- Foreclosures and rights transfers priced far below market value
- Duplicate listings within and across portals
- Fake locations (developments in other municipalities listed as GAM neighborhoods)
- Data entry errors (impossible surface areas, construction year entered as age)
- Inconsistent neighborhood names

## Approach

1. **Integration:** listings are merged into one record per house when price (±1%), built m² (±5%) and neighborhood (or distance < 600 m) match, using union-find.
2. **Cleaning:** rule-based exclusion of foreclosures, non-residential listings, fake locations and outliers, with every exclusion logged.
3. **Comparable features:** like a real appraiser, the model uses the smoothed price/m² of the neighborhood and of the 10 nearest homes (BallTree, haversine distance), computed with **cross-fitting** to prevent data leakage.
4. **Models:** Ridge, Random Forest and Gradient Boosting, each with and without comparables, against two non-ML baselines.
5. **Validation:** 5-fold cross-validation × 3 seeds, plus GroupKFold by neighborhood to test on unseen areas.

## Results

Metrics on total price, out-of-sample:

| Model | Median error | Within ±15% | Within ±25% | MAPE |
|---|---|---|---|---|
| **Random Forest + comparables** | **21.2%** | **37.6%** | **56.7%** | **32.3%** |
| Random Forest (no comparables) | 22.6% | 33.8% | 54.5% | 32.5% |
| Gradient Boosting (no comparables) | 22.9% | 33.8% | 53.6% | 33.7% |
| Ridge + comparables | 23.4% | 33.9% | 52.9% | 34.0% |
| Baseline: neighborhood median | 28.5% | 30.0% | 46.3% | 40.5% |
| Baseline: borough median | 29.1% | 25.7% | 43.6% | 40.7% |

The best model cuts the median error by **7.3 points** compared with the best non-ML baseline. It also outputs a calibrated **80% prediction interval** and a confidence flag when there are few comparable homes nearby.

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Place the three listing CSVs in a `data/` folder (or upload them when running in Google Colab):
   - `casas_gam_mercadolibre.csv`
   - `casas_gam_vivanuncios.csv`
   - `casas_gam_lamudi.csv`
3. Run all cells of `PREDICTOR-PRECIO-CASAS-GAM-CDMX.ipynb`.

## Limitations

- Prices are asking prices, not closing prices.
- Few listings in lower-income neighborhoods, so intervals are wider there.
- Listings rarely describe the condition of the house.

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Google Colab

## Author

**Héctor Giovanni García Alvarado** — AI Engineering student, ESCOM-IPN
