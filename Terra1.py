# # requirements.txt
# torch
# git+https://github.com/openai/CLIP.git
# numpy
# matplotlib
# Pillow
# streamlit

# app.py (Streamlit code)
import streamlit as st
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def main():
    st.title("CLIP-Based Classification App")

    device = "cpu"  # or "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "jfif"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        image = preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize(["Wildfire Fire", "No Wildfire Fire"]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        labels = ["Yes", "No"]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(labels, probs, color='gray')
        ax.set_title("Probabilities")
        ax.set_ylim([0, 1])
        for i, v in enumerate(probs):
            ax.text(i, v, f"{v:.3f}", ha='center', va='bottom')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
