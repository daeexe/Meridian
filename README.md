# Meridian Reach and Frequency Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)  
[View source on GitHub](https://github.com/google/meridian)

---

## 📖 Overview
This repository provides an end-to-end **demo** showcasing the fundamental functionalities of **[Google Meridian](https://github.com/google/meridian)** for **Reach and Frequency (RF) data**.  

It includes working examples of the major modeling steps:  
- Install dependencies  
- Load and clean data  
- Configure the model with ROI priors  
- Run model diagnostics  
- Generate model results & two-page HTML output  
- Run budget optimization & export optimization report  
- Save and reload the model object  

⚠️ **Note:** This demo uses **sample data** and skips exploratory data analysis (EDA) and preprocessing. Real-world datasets will require additional steps.

---

## 🚀 Getting Started

### Step 0: Install
1. Use a **GPU runtime** in Google Colab (recommended: T4 GPU).  
   - Change runtime in Colab: `Runtime > Change runtime type`  

2. Install the latest version of Meridian:  
```bash
pip install --upgrade google-meridian[colab,and-cuda]
```

3. Verify GPU and CPU availability:
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
```

---

### Step 1: Load & Clean Data
- Mount Google Drive in Colab.  
- Load raw CSV dataset.  
- Clean column names, numeric columns, and date fields.  
- Rename fields to match Meridian format.  
- Aggregate by **Week** and **Region_(Matched)**.  
- Save the cleaned dataset to CSV:  
```python
output_path = '/content/drive/MyDrive/cleaned_meridian.csv'
pivot_table.to_csv(output_path, index=False)
```

---

### Step 2: Configure Model
Meridian uses **Bayesian inference** and **MCMC** sampling.  
- Define ROI priors with a LogNormal distribution.  
- Initialize and fit the model:
```python
from meridian.model import model, spec, prior_distribution
import tensorflow_probability as tfp

roi_rf_mu = 0.2
roi_rf_sigma = 0.9
prior = prior_distribution.PriorDistribution(
    roi_rf=tfp.distributions.LogNormal(roi_rf_mu, roi_rf_sigma)
)
model_spec = spec.ModelSpec(prior=prior)
mmm = model.Meridian(input_data=data, model_spec=model_spec)

mmm.sample_prior(500)
mmm.sample_posterior(n_chains=10, n_adapt=2000, n_burnin=500, n_keep=500, seed=1)
```

---

### Step 3: Run Diagnostics
- Convergence check (R-hat < 1.2):  
```python
from meridian.analysis import visualizer
model_diagnostics = visualizer.ModelDiagnostics(mmm)
model_diagnostics.plot_rhat_boxplot()
```

- Model fit (expected vs. actual sales):  
```python
model_fit = visualizer.ModelFit(mmm)
model_fit.plot_model_fit()
```

---

### Step 4: Generate Model Results
Export **two-page HTML summary output**:
```python
from meridian.analysis import summarizer
mmm_summarizer = summarizer.Summarizer(mmm)

filepath = '/content/drive/MyDrive'
mmm_summarizer.output_model_results_summary(
    'summary_output.html', filepath, '2021-01-25', '2024-01-15'
)
```

---

### Step 5: Budget Optimization
Optimize media allocation under a **fixed budget scenario**:
```python
from meridian.analysis import optimizer
budget_optimizer = optimizer.BudgetOptimizer(mmm)
optimization_results = budget_optimizer.optimize()

optimization_results.output_optimization_summary(
    'optimization_output.html', filepath
)
```

---

### Step 6: Save & Reload Model
Save model for future use:
```python
file_path='/content/drive/MyDrive/saved_mmm.pkl'
model.save_mmm(mmm, file_path)
```

Reload model:
```python
mmm = model.load_mmm(file_path)
```

---

## 📌 Notes
- The demo uses **simulated data**; results will differ with real-world datasets.  
- For details on model calibration, diagnostics, and custom optimization, see the official [Meridian documentation](https://github.com/google/meridian).  

---

## 📂 Repository Structure
```
├── README.md               <- Project documentation
├── demo_notebook.ipynb     <- Colab-ready demo notebook
├── data/
│   └── MMM_Meridian.csv    <- Sample dataset (replace with your own)
└── outputs/
    ├── cleaned_meridian.csv
    ├── summary_output.html
    └── optimization_output.html
```
---

## 📎 Usage

This repository is intended for **research and educational purposes only**.

While the code is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0), which permits commercial use, **I do not endorse or support** any commercial usage of this modified version without prior written permission.

If you are planning to use this project for commercial applications (e.g. as part of a product, service, or for client work), please [contact me](mailto:your.email@example.com) first.

By using this repository, you agree to:

- Properly attribute the original authors ([Google's Meridian project](https://github.com/google/meridian))
- Avoid using this code or outputs for direct commercial gain without permission
- Respect the spirit of open collaboration and responsible AI development

---

## 📜 License
This project is distributed under the **Apache 2.0 License**.  
See the [LICENSE](LICENSE) file for more details.
