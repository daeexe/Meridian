readme_content = """# Meridian Reach and Frequency Demo

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
