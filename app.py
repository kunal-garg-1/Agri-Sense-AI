import pathlib
import platform

# Smart Cross-Platform Path Hack (Works on both Windows PC and Linux Cloud!)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        import pathlib._local
        pathlib._local.PosixPath = pathlib._local.WindowsPath
    except ImportError:
        pass

import streamlit as st
import pandas as pd
import joblib
import torch
import torchvision.transforms as T
from fastai.vision.all import load_learner
from PIL import Image

# 2. DRAW THE UI FIRST
st.set_page_config(page_title="Agri-Sense", layout="wide")
st.title("Agri-Sense: AI Farming Assistant 🌱")

# 3. LOAD MODELS SAFELY 
@st.cache_resource
def load_engines():
    # Load Tabular Price Engine
    p_model = joblib.load('market_price_model.pkl')
    m_cols = joblib.load('model_columns.pkl')
    
    # Load Vision Engine (Extracting pure PyTorch to bypass fast.ai dictionary bugs)
    learner = load_learner('crop_disease_model.pkl')
    v_model = learner.model.eval()
    v_vocab = learner.dls.vocab
    
    return p_model, m_cols, v_model, v_vocab

# The UI will show this spinner while the heavy models load in the background
with st.spinner("Waking up the AI Engines... (This may take 15-30 seconds on CPU)"):
    price_model, model_cols, vision_model, vocab = load_engines()

# 4. SIDEBAR MENU
menu = st.sidebar.selectbox("Select Feature", ["Crop Disease Diagnosis", "Market Price Predictor"])

# --- FEATURE 1: Disease Diagnosis (Pure PyTorch Bypass) ---
if menu == "Crop Disease Diagnosis":
    st.header("📸 Upload a Leaf Photo")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption='Uploaded Leaf', use_container_width=True)
            
        with col2:
            st.info("Analyzing Plant Pathology...")
            
            # Pure PyTorch Image Pipeline
            transform = T.Compose([
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                preds = vision_model(img_tensor)
                probs = torch.softmax(preds, dim=1)[0]
                confidence, pred_idx = torch.max(probs, dim=0)
            
            diagnosis = vocab[pred_idx.item()]
            conf_val = confidence.item()
            
            st.metric(label="Diagnosis", value=diagnosis.replace("_", " ").title())
            st.progress(conf_val, text=f"Confidence: {conf_val:.2%}")

# --- FEATURE 2: Price Predictor ---
elif menu == "Market Price Predictor":
    st.header("📈 Predict Future Crop Rates")
    
    col1, col2 = st.columns(2)
    with col1:
        crop_type = st.text_input("Enter Crop Name (e.g., Maize, Wheat):")
        state = st.text_input("Enter State Name (e.g., India, Punjab):")
    
    if st.button("Calculate Market Price"):
        input_data = pd.DataFrame({'commodity_name': [crop_type.title()], 'state_name': [state.title()]})
        input_encoded = pd.get_dummies(input_data)
        
        input_encoded = input_encoded.reindex(columns=model_cols, fill_value=0)
        
        predicted_price = price_model.predict(input_encoded)[0]
        st.success(f"Estimated Modal Price: ₹{predicted_price:.2f} per Quintal")