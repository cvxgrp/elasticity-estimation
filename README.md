# Estimating Price Elasticity Matrices

This repository implements a method for estimating the **price elasticity matrix**
of a set of products from historical price and demand data. For theoretical background,
please refer to our [manuscript](https://stanford.edu/~boyd/papers/elasticity_estimation.html).
If you use this code, please cite
```
@article{SB26,
    author={Maximilian Schaller and Stephen Boyd},
    title={Estimating Price Elasticity Matrices},
    note = {available on arXiv at \url{https://arxiv.org/pdf/2604.XXXXX}},
    year={2026}
}
```

## Data requirements

### Option A — Dominick's Finer Foods (DFF) scanner data

Place the following raw CSV files from the
[DFF data set](https://www.chicagobooth.edu/research/kilts/research-data/dominicks)
in the project root:

| File | Description |
|------|-------------|
| `wber.csv` | DFF beer category weekly scanner data |
| `wbjc.csv` | DFF bottled juice category weekly scanner data |
| `wsdr.csv` | DFF soft drinks category weekly scanner data |

Then run the preprocessing script to generate `demand.csv` and `prices.csv`:

```bash
python dff_data.py
```

### Option B — Generic demand/price data

Provide the following two CSV files directly (no header row, no index column):

| File | Shape | Description |
|------|-------|-------------|
| `demand.csv` | `(times, products)` | Integer unit sales per product per time interval |
| `prices.csv` | `(times, products)` | Price per unit and product per time interval (positive) |

## Scripts

| Script | Description |
|--------|-------------|
| `dff_data.py` | Preprocess DFF scanner data → `demand.csv`, `prices.csv` |
| `figure4_5.py` | Synthetic-data experiment: log-likelihood CV + elasticity heatmaps |
| `figure6.py` | Synthetic-data experiment: pricing-performance CV |
| `figure7.py` | Real-data experiment (requires `demand.csv`/`prices.csv`): log-likelihood CV |

To reproduce the figures in our manuscript:

```bash
python figure4_5.py
python figure6.py
python figure7.py   # requires demand.csv and prices.csv
```

## Module structure

```
estimation.py           Core estimation: gradient ascent (GA), alternating
                        maximization (AM), and nonlinear programming (NLP)
evaluation.py           Log-likelihood, prediction error, pricing performance,
                        and K-fold cross-validation
optimal_pricing/        Submodule: product-pricing problem (PPP)
```

## Dependencies

```
numpy
pandas          (dff_data.py only)
cvxpy
ipopt           (figure6.py and when using NLP only)
matplotlib      (figure4_5.py figure6.py figure7.py only)
```
