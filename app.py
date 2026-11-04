import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import os

st.set_page_config(page_title="Agri-Sense", layout="wide")
st.title("Agri-Sense: AI Farming Assistant 🌱")

@st.cache_resource
def load_engines():
    # 1. Load Tabular Price Engine
    p_model = joblib.load('market_price_model.pkl')
    m_cols = joblib.load('model_columns.pkl')
    
    # 2. Load Pure PyTorch Vision Engine (No fastai needed!)
    v_vocab = joblib.load('disease_vocab.pkl')
    
    # Rebuild the exact Fast.ai ResNet18 architecture in pure PyTorch
    from fastai.vision.all import create_vision_model, resnet18
    v_model = create_vision_model(resnet18, n_out=len(v_vocab), pretrained=False)
    
    # Load weights safely across any PyTorch version
    state_dict = torch.load('disease_weights.pth', map_location='cpu', weights_only=False)
    v_model.load_state_dict(state_dict)
    v_model.eval()
    
    return p_model, m_cols, v_model, v_vocab

# Diagnostic check in case GitHub LFS blocked the file download on Cloud
if os.path.exists('disease_weights.pth') and os.path.getsize('disease_weights.pth') < 10000:
    st.error("⚠️ GitHub LFS Pointer Detected! Streamlit Cloud downloaded the 130-byte LFS pointer instead of the 43MB model.")
    st.stop()

with st.spinner("Waking up the AI Engines..."):
    price_model, model_cols, vision_model, vocab = load_engines()

menu = st.sidebar.selectbox("Select Feature", ["Crop Disease Diagnosis", "Market Price Predictor"])

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
            transform = T.Compose([
                T.Resize((128, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                preds = vision_model(img_tensor)
                probs = torch.softmax(preds, dim=1)[0]
                confidence, pred_idx = torch.max(probs, dim=0)
            
            diagnosis = vocab[pred_idx.item()]
            conf_val = confidence.item()
            
            st.metric(label="Diagnosis", value=diagnosis.replace("_", " ").title())
            st.progress(conf_val, text=f"Confidence: {conf_val:.2%}")

elif menu == "Market Price Predictor":
    st.header("📈 Predict Future Crop Rates")
    col1, col2 = st.columns(2)
    with col1:
        crop_type = st.text_input("Enter Crop Name (e.g., Maize, Wheat):")
        state = st.text_input("Enter State Name (e.g., India, Punjab, Rajasthan):")
    
    if st.button("Calculate Market Price"):
        input_data = pd.DataFrame({'commodity_name': [crop_type.title()], 'state_name': [state.title()]})
        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=model_cols, fill_value=0)
        
        predicted_price = price_model.predict(input_encoded)[0]
        st.success(f"Estimated Modal Price: ₹{predicted_price:.2f} per Quintal")