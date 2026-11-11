# 🌱 Agri-Sense: Multi-Modal AI Ecosystem for Precision Agriculture

[![Live Demo](https://img.shields.io/badge/🚀_Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://agri-sense-ai-tool.streamlit.app/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24125244.svg)](https://doi.org/10.5281/zenodo.24125244)

**Agri-Sense** is an end-to-end, dual-engine Machine Learning application designed to solve two critical bottlenecks in agricultural production: **real-time plant pathology diagnosis** and **commodity market price volatility**. 

Rather than relying on external third-party API wrappers, Agri-Sense runs two distinct, locally serialized mathematical architectures simultaneously—bridging Deep Learning Computer Vision with Classical Ensemble Regression.

🔗 **Live Deployment:** [https://agri-sense-ai-tool.streamlit.app/](https://agri-sense-ai-tool.streamlit.app/)

📄 **Published Research Preprint (CERN / Zenodo):** [AgriSense AI: An Integrated Dual-Engine Deep Transfer Learning and Ensemble Regression Framework (DOI: 10.5281/zenodo.24125244)](https://doi.org/10.5281/zenodo.24125244)

> **Citation:** K. Garg, *"AgriSense AI: An Integrated Dual-Engine Deep Transfer Learning and Ensemble Regression Framework for Crop Foliar Disease Diagnosis and Agricultural Commodity Price Forecasting,"* Zenodo (CERN), Nov. 2026. DOI: `10.5281/zenodo.24125244`

---

## 🧠 System Architecture (Dual-Engine Pipeline)

### 1. Computer Vision Engine (Plant Pathology Diagnosis)
* **Backbone Architecture:** Pre-trained `ResNet18` Convolutional Neural Network (CNN) fine-tuned via Transfer Learning.
* **Dataset:** Trained on the standardized *PlantVillage* dataset across **15 distinct crop-pathology classes** (Tomato Early/Late Blight, Potato Blight, Pepper Bacterial Spot, etc.), normalized and resized to `128x128` RGB tensors.
* **Validation Metrics:** Achieved **99.97%–100% validation accuracy** (`0.000490` validation loss) across a stratified 80/20 train-validation split using T4 GPU acceleration.

### 2. Tabular Regression Engine (Commodity Price Forecasting)
* **Algorithm:** 100-Tree `RandomForestRegressor` ensemble trained on historical Indian agricultural market (*Mandi*) data.
* **Feature Engineering & Leakage Prevention:** Stripped target-leaking features (`min_price`, `max_price`) and applied One-Hot Encoding (`pd.get_dummies`) across 17 dimensional features representing regional state markets and crop commodities.
* **Evaluation Metrics:** Achieved an **$R^2$ score of 0.7200** (explaining 72% of historical market variance) and a **Mean Absolute Error (MAE) of ₹1,450.25** per quintal across volatile multi-year pricing spreads.

---

## 🛠️ Production Engineering & MLOps Challenges Solved

Deploying deep learning models alongside tabular ensembles across heterogeneous environments (Linux training clusters, Windows local testing, Linux cloud containers) presents severe serialization conflicts. Key engineering solutions implemented in `app.py` include:

1. **Pure PyTorch State-Dict Decoupling (`disease_weights.pth`):** 
   High-level `fast.ai` `.pkl` exports tightly couple Python version internals (`3.12` vs `3.13`/`3.14`) and OS pathing classes (`PosixPath` vs `WindowsPath`), causing `AttributeError` and `UnboundLocalError` crashes during cloud unpickling. To make inference completely OS- and version-agnostic, the raw PyTorch `state_dict` and class vocabulary were decoupled from the learner wrapper and reconstructed natively in PyTorch at runtime.
2. **Custom Torchvision Tensor Pipeline:** 
   Bypassed high-level prediction wrappers by routing uploaded `PIL` images directly through a deterministic `torchvision.transforms` pipeline (`Resize(128, 128)` to `ToTensor()` to `ImageNet Normalization` to `Softmax` probability extraction).
3. **Zero-LFS Weight Packaging & RAM Caching:** 
   Eliminated Git LFS bandwidth bottlenecks on cloud containers by serializing pure tensor weights (`~44MB`) directly into standard Git history and caching both engines in server RAM via `@st.cache_resource` for millisecond inference latency.

---

## 📂 Repository Structure

```text
Agri-Sense-AI/
├── app.py                     # Multi-modal Streamlit inference server
├── disease_weights.pth        # Serialized ResNet18 PyTorch state_dict
├── disease_vocab.pkl          # 15-class PlantVillage pathology vocabulary
├── market_price_model.pkl     # Trained 100-Tree Random Forest Regressor
├── model_columns.pkl          # One-Hot Encoded feature schema (17 dimensions)
├── requirements.txt           # Production dependencies
└── README.md                  # System documentation
```

💻 Run Locally
1. Clone the repository:

``` Bash
git clone [https://github.com/kunal-garg-1/Agri-Sense-AI.git](https://github.com/kunal-garg-1/Agri-Sense-AI.git)
cd Agri-Sense-AI
```

2. Install dependencies:
```
Bash
pip install -r requirements.txt
```

3. Launch the application:
```
Bash
streamlit run app.py
```

