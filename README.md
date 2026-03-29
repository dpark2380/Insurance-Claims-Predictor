# Insurance Claims Severity and Frequency Modelling

## Project Overview

This project builds a **pure premium estimation framework** for motor insurance using the French Motor Third-Party Liability (MTPL) dataset. The goal is to predict the expected cost of insuring a given policy by modelling two separate but related questions:

1. **Frequency:** Given a policy, what is the probability that a claim occurs?
2. **Severity:** Given that a claim occurs, how much will it cost?

The pure premium is the product of these two estimates:

```
Pure Premium = P(claim) x E(claim amount | claim occurred)
```

This two-stage approach mirrors how insurance pricing actually works in practice. Actuaries separate frequency and severity because the drivers of each are different — a high-risk driver (high BonusMalus) claims more often, but when a claim occurs the repair bill is not necessarily larger than a careful driver's.

---

## Why This Project

Insurance pricing is one of the most data-rich and statistically rigorous applications of predictive modelling. In APRA-regulated markets like Australia, pricing models must be both accurate and explainable to regulators. This creates a fundamental tension between model complexity and interpretability that does not exist in the same way in most other industries.

This project directly addresses that tension by benchmarking three modelling approaches of increasing complexity:

- **Gamma/Poisson GLM** — the actuarial industry standard, fully interpretable, coefficients map directly to pricing relativities
- **XGBoost** — a gradient boosted tree model that captures non-linear feature interactions automatically
- **Neural Network (PyTorch)** — a feedforward network, the most flexible approach but least interpretable

At the end of the project, all three models are combined into a single premium estimator that takes a policy as input and returns predictions from each model side by side. This allows a direct, quantified comparison of the accuracy-interpretability tradeoff.

---

## Dataset

**Source:** French Motor Third-Party Liability dataset (freMTPL2freq and freMTPL2sev), publicly available via OpenML. This is a standard actuarial benchmark used in academic literature and industry research.

**Size:** 679,513 policies, of which 26,439 resulted in at least one claim (3.9% claim rate).

**Features:**

| Feature | Type | Description |
|---|---|---|
| BonusMalus | Numeric | Claims history score. 50 = clean record, increases with each claim. The most predictive variable in motor pricing. |
| DrivAge | Numeric | Age of the driver in years |
| VehAge | Numeric | Age of the vehicle in years |
| VehPower | Numeric | Engine power of the vehicle |
| Density | Numeric | Population density of the policyholder's area (people per km²) |
| Area | Categorical | Geographic risk zone (A to F) |
| VehBrand | Categorical | Anonymised vehicle brand code |
| VehGas | Categorical | Fuel type (Diesel or Regular) |
| Region | Categorical | French administrative region (22 categories) |
| Exposure | Numeric | Fraction of the year the policy was active (used as offset in frequency model) |

**Target variables:**
- `ClaimNb` — number of claims (frequency model target)
- `ClaimAmount` — total claim amount in EUR, capped at the 99th percentile of 16,451 EUR (severity model target)

---

## Project Structure

```
insurance_claims_severity/
├── data/
│   ├── freMTPL2freq.csv        # Raw frequency dataset (679k policies)
│   ├── freMTPL2sev.csv         # Raw severity dataset (26k claims)
│   ├── claims_cleaned.csv      # Cleaned severity dataset for modelling
│   └── policies_cleaned.csv    # Cleaned frequency dataset for modelling
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis (frequency and severity)
│   ├── 02_glm.ipynb            # Gamma GLM (severity) and Poisson GLM (frequency)
│   ├── 03_xgboost.ipynb        # XGBoost models for both frequency and severity
│   ├── 04_neural_net.ipynb     # PyTorch neural network for both
│   └── 05_comparison.ipynb     # Model comparison and pure premium estimator
├── models/                     # Saved model files
├── results/                    # Output charts and metric tables
└── README.md
```

---

## Modelling Approach

### Frequency Model

**Target:** Number of claims per policy (ClaimNb)
**Distribution:** Poisson — appropriate for non-negative integer count data
**Link function:** Log — ensures predictions are always positive
**Exposure offset:** log(Exposure) is included as an offset to account for varying policy durations. A policy active for 6 months should have half the predicted claim count of an otherwise identical 12-month policy.

### Severity Model

**Target:** Claim amount in EUR given a claim occurred (ClaimAmount)
**Distribution:** Gamma — appropriate for positive, right-skewed continuous data
**Link function:** Log — ensures positive predictions and makes the model multiplicative (each feature multiplies the expected claim amount by a factor)
**Why not OLS:** Ordinary linear regression assumes normally distributed errors and can produce negative predictions. Both assumptions are violated for claim amounts.

### Data Preprocessing

- Claims above the 99th percentile (16,451 EUR) are capped. Catastrophic claims are typically handled separately by reinsurance and are not representative of the pricing model's intended scope.
- 5 records with placeholder values (VehAge or DrivAge = 99) are removed.
- Categorical features are one-hot encoded with drop_first=True to avoid multicollinearity.
- Exposure is split alongside features and target to maintain alignment across the train/test split.

---

## Key Findings (in progress)

### EDA

- Claim amounts are heavily right-skewed (mean 1,569 EUR vs median 1,172 EUR). Many claims cluster at round amounts (1,204 EUR, 1,128 EUR) consistent with standardised repair tariffs.
- Individual features show near-zero R² against claim amount. This is expected — features predict whether a claim happens, not how much it costs.
- Frequency shows much clearer patterns. BonusMalus is the strongest predictor, with drivers scoring 150+ claiming at 24% vs 3% for clean-record drivers (50-60).
- Young drivers (18-25) have a claim rate of 6.8%, dropping sharply into the mid-twenties and rising modestly for the 65+ group.

### GLM

- The Gamma GLM achieves MAE of 1,079 EUR and Gamma Deviance of 1.03 on the severity test set.
- The Poisson GLM achieves MAE of 0.077 claims and Poisson Deviance of 46,581 on the frequency test set.
- The GLM correctly captures the direction of BonusMalus and DrivAge effects but compresses predictions toward the mean and misses non-linear patterns — particularly the elevated claim rate for young and older drivers.
- These limitations set the baseline that XGBoost and the neural network are benchmarked against.

### XGBoost and Neural Network

*(Results to be added)*

### Model Comparison

*(Results to be added)*

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd insurance_claims_severity

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch torchvision jupyter statsmodels
```

Run notebooks in order from 01 to 05. The first notebook downloads the dataset automatically from OpenML — no manual download required.

---

## Dependencies

- Python 3.11+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- statsmodels
- xgboost
- torch (PyTorch)
- jupyter

---

## Business Context

In motor insurance, the pure premium is the theoretical cost of providing coverage to a single policy before expenses and profit margin are added. Pricing actuaries build models to estimate this for every new policy that comes in, using the characteristics of the vehicle and driver as inputs.

Getting this right matters because:
- Overpricing high-risk policies loses customers to competitors
- Underpricing high-risk policies leads to losses
- Regulators (such as APRA in Australia) require that pricing models can be explained and justified

This project demonstrates the full pipeline from raw data to a deployable premium estimate, including the model selection decisions and tradeoffs that a pricing actuary would face in practice.
