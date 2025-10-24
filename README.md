# Amazon ML Challenge 2025: Smart Product Pricing (AIR 12)

This repository contains the code and methodology for our **12th Place (All India Rank)** solution in the Amazon ML Challenge 2025: Smart Product Pricing.

Our solution implements a two-stage hybrid architecture that leverages a fine-tuned Vision-Language Model (VLM) for powerful feature extraction and an XGBoost model for robust price regression.

## 🏆 Team Bezzzos

* **Team Name:** Bezzzos
* **Team Members:** Mukil M, Priyanshu Kumar, Suryansh Mishra, Dilshad Raza

---

## 🚀 Solution Overview

Our approach is a **two-stage, feature-extraction-based model** that combines deep learning with traditional machine learning to achieve strong predictive accuracy.

1.  **Stage 1: Price-Aware Embedding Extraction**
    We first fine-tune the **Qwen-2.5-VL-Instruct-7B** Vision-Language Model using a PEFT (LoRA) strategy. The VLM is trained to process both product images and text to produce rich, price-aware embeddings.

2.  **Stage 2: Gradient Boosting Regression**
    The embeddings generated in Stage 1 are saved and then used as the input features for an **XGBoost Regressor**. This model is trained to predict the final product price based on the VLM-generated features.

Our core innovation was **decoupling feature engineering from regression**. Instead of using the VLM for end-to-end prediction, we leverage its power to create superior, context-aware features. This allows the XGBoost model to focus solely on learning the complex relationships between these features and the price, leading to better performance and faster iteration.

---

## 💡 Key Challenges & Strategy

Our methodology was designed to address several key challenges in the dataset:

* **SMAPE Metric Behavior:** The SMAPE metric heavily penalizes errors on low-priced items more than on high-priced ones. This required a model that was highly accurate across the entire price spectrum.
* **Price Distribution:** The product prices had a significant right skew. To stabilize model training, we applied a **logarithmic transformation (`np.log1p`)** to the target variable.
* **Loss Function:** We used **L1 Loss (Mean Absolute Error)** during VLM fine-tuning. Its constant gradient helps prevent exploding gradients and ensures steady convergence, which is particularly effective for log-transformed values.

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
    * **Precision:** bfloat16 / float16 (mixed precision)
* **Output:** After fine-tuning, the model was used as a feature extractor to process the entire dataset, generating a single embedding vector from the final hidden state for each product.

### Stage 2: Gradient Boosting Regression (XGBoost
* **Model:** XGBoost Regressor
* **Input:** The price-aware embeddings generated from Stage 1.
* **Output:** The log-transformed price. This output is converted back to the original price scale using `np.expm1()` for final evaluation.
* **Key Hyperparameters:**
    * **Objective:** `reg:absoluteerror`
    * **Number of Estimators:** 7500
    * **Learning Rate:** 0.05
    * **Max Depth:** 5
    * **Subsample:** 0.8
    * **Column Subsample (colsample_bytree):** 0.8
    * **Evaluation Metric:** `mae`

---

## 📊 Model Performance

The final performance was evaluated on a 20% hold-out validation set. The scores below are for the final XGBoost model predictions on the **original price scale**.

| Metric | Score |
| :--- | :--- |
| **SMAPE Score** | 41.6577% |
| **Mean Squared Error (MSE)** | 889.0822 |

---

## 🏁 Conclusion

Our solution successfully demonstrates the power of a hybrid deep learning and traditional machine learning approach. By leveraging a fine-tuned vision-language model to create high-dimensional, price-aware embeddings, we provided our XGBoost model with features that captured complex, multimodal nuances.

This architecture highlights the effectiveness of using large models for **intelligent feature engineering** rather than as monolithic, end-to-end predictors.
