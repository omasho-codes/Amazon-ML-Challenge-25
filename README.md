
# [Amazon ML Challenge 2025 ](https://docs.google.com/spreadsheets/d/e/2PACX-1vSA000odhpNXaEJTqAdDbwSj3T_fjH-w7QTCrCPIP54VL0bQUTVvok4hVFRpPCwsYTXY0BAvEUDQ2XS/pubhtml): Smart Product Pricing (AIR 12)

This repository contains the code and methodology for our **12th Place (All India Rank)** solution in the Amazon ML Challenge 2025: Smart Product Pricing.

Our solution implements a two-stage hybrid architecture that leverages a fine-tuned Vision-Language Model (VLM) for powerful feature extraction and an grand ensemble of XGBoost, LightGBM and Catboost (4x each) models for robust price regression.

## 🏆 Team Bezzzos

* **Team Name:** Bezzzos
* **Team Members:** Suryansh Mishra, Mukil M, Priyanshu Kumar, Dilshad Raza

---

## 🚀 Solution Overview

Our approach is a **two-stage, feature-extraction-based model** that combines rich Qwen's Image + Text embeddings with multiple Tree-based GBMs.

1.  **Stage 1: Price-Aware Embedding Extraction**
    We first fine-tune the **Qwen-2.5-VL-Instruct-7B** Vision-Language Model using a PEFT (LoRA) with a MLP head which is optmised for L1 loss on log of prices. The VLM is trained to process both product images and text to produce large information rich embeddings.

2.  **Stage 2: Gradient Boosting Regression**
    The embeddings generated in Stage 1 are saved and then used as the input features for **Tree-based Regressors**. This model is trained to predict the final product price based on the VLM-generated features.

---

## 💡 Key Challenges & Strategy

Our methodology was designed to address several key challenges in the dataset:

* **SMAPE Metric Behavior:** The SMAPE metric heavily penalizes errors on low-priced items more than on high-priced ones. This required a model that was highly accurate, specfically for lower price range, so to not give diminishing gradients when being too close to actual price.
* **Price Distribution:** The product prices had a significant right skew. To stabilize training numerically and align our loss with the **SMAPE METRIC**, we applied a **logarithmic transformation (`np.log1p`)** to the target variable, this made it more like a normal distribution.
* **Loss Function:** We used **L1 Loss (Mean Absolute Error)** during VLM fine-tuning. Its constant gradient helps prevent vanishing gradients when being close to predicted prices for lower price ranges.

Summary -
These improvements boosted our fine-tuning by many folds on the public LB and also aligning the loss with the SMAPE metric.

---

## 🛠️ Model Architecture

### Stage 1: Price-Aware Embedding Extraction (VLM)

* **Base Model:** `Qwen/Qwen2.5-VL-7B-Instruct`
* **Input Prompt Structure:** The model was fed a prompt combining the product image and its catalog text.
    ```
    <img>{image}</img>
    Catalog: {catalog_content}
    Instruction: Encode product for price prediction.
    ```
* **Data Preprocessing:**
    * Images were resized to a standard 224x224 resolution.
    * The `price` target variable was transformed using `np.log1p()`.
* **Fine-Tuning Details:**
    * **PEFT Method:** LoRA
    * **LoRA Rank (r):** 16
    * **LoRA Alpha:** 32
    * **LoRA Target Modules:** `"q_proj"`, `"v_proj"`, `"k_proj"`, `"o_proj"`
    * **Regression Head:** A custom two-layer MLP (`Linear (4096, 512) -> ReLU -> Linear(512, 1)`) was added to the VLM for the price prediction task during fine-tuning.
    * **Optimizer:** AdamW
    * **Learning Rate:** 5e-5
    * **Loss Function:** L1Loss (MAE)
    * **Epochs:** 3
    * **Precision:** bfloat16 / float32 (mixed precision)
* **Output:** After fine-tuning, the model was used as a feature extractor to process the entire dataset, generating a single 4096-dimensional embedding vector from the final hidden state for each product.

### Stage 2: Ensemble of Gradient Boosting Models

* **Models:** An ensemble of **XGBoost**, **CatBoost**, and **LightGBM**.
* **Input:** The 4096-dimensional embeddings generated from Stage 1.
* **Output:** The log-transformed price (`np.log1p(price)`). This output is converted back to the original price scale using `np.expm1()` for final evaluation.

#### Ensemble Strategy: Hierarchical Feature-Subset Averaging

Our final ensemble consists of 12 models (4 for each GBM type). Instead of training each model on all 4096 features, we used a feature-subset averaging technique to improve robustness and reduce model correlation.

1.  **Feature Importance:** We first trained an initial model (XGB, Cat, LGBM) on the *full* feature set to generate a feature importance list specific to each model type.
2.  **Subset Training:** For each model type, we defined four feature-set sizes: **[Top 500, Top 1000, Top 2000, Top 3500]**. We then trained 4 separate models, one for each of these "top N" feature subsets.
3.  **Model-Type Averaging:** The predictions from these 4 subset-models were averaged to create a single, robust prediction for that model type (e.g., "Averaged XGBoost").
4.  **Grand Ensemble:** The final prediction is the simple average of the three "Averaged" model predictions (Averaged XGB + Averaged Cat + Averaged LGBM) / 3.

#### Key Hyperparameters

All models were trained to optimize for MAE (L1 loss) on the log-transformed price, as this aligned well with the final SMAPE evaluation metric.

**1. XGBoost (4 models)**
* **Objective:** `reg:absoluteerror`
* **Number of Estimators:** 2000
* **Learning Rate:** 0.05
* **Max Depth:** 5
* **Subsample:** 0.8
* **Column Subsample (colsample_bytree):** 0.8
* **Evaluation Metric:** `mae`
* **Early Stopping:** 50 rounds

**2. LightGBM (4 models)**
* **Objective:** `mae`
* **Number of Estimators:** 2000
* **Learning Rate:** 0.05
* **Max Depth:** 5
* **Subsample:** 0.8
* **Column Subsample (colsample_bytree):** 0.8
* **Evaluation Metric:** Custom SMAPE (calculated on `np.expm1` values)
* **Early Stopping:** 100 rounds

**3. CatBoost (4 models)**
* **Loss Function:** `MAE`
* **Iterations:** 2000
* **Learning Rate:** 0.05
* **Depth:** 5
* **Subsample:** 0.8
* **Column Subsample (colsample_bylevel):** 0.8
* **Evaluation Metric:** `MAE`
* **Early Stopping:** 100 rounds

---

## 📊 Model Performance

The final performance was evaluated on a 20% hold-out validation set. The scores below are for the final grand ensemble model predictions on the **original price scale**.

| Metric | Score |
| :--- | :--- |
| **SMAPE Score** | 41.6577% |
| **Mean Squared Error (MSE)** | 889.0822 |

---

## 🏁 Conclusion

Our solution successfully demonstrates the power of a hybrid deep learning and traditional machine learning approach with loss choices and preprocesing. By leveraging a fine-tuned vision-language model to create high-dimensional, information dense features which we provided to our GBMs with features that captured complex, multimodal nuances.

This architecture highlights the effectiveness of using large models for **intelligent feature engineering** rather than as monolithic, end-to-end predictors.


<img src="https://github.com/user-attachments/assets/a99712f3-f86b-4e4b-9d49-2f07c845fbdd" width="600"/>

