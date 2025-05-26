import sys, types, os
sys.modules['torch.classes'] = types.SimpleNamespace()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["STREAMLIT_WATCH_SYSTEM_FOLDERS"] = "false"

import streamlit as st
import torch
import numpy as np
import pickle
from torchvision import models, transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import gzip
from huggingface_hub import hf_hub_download

st.set_page_config(layout="wide")

st.markdown("""
<style>
/* Widen content area */
.block-container {
    padding: 3rem 5rem;
    max-width: 95%;
}

/* Headings */
h1 {
    font-size: 2.5rem !important;
    margin-bottom: 0.75rem;
}
h2 {
    font-size: 2rem !important;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
}
h3 {
    font-size: 1.5rem !important;
    margin-bottom: 0.5rem;
}

/* Paragraphs */
p, li {
    font-size: 1.15rem !important;
    line-height: 1.7;
    margin-bottom: 1rem;
    text-align: left;
}

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    font-size: 1rem !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    color: #21c6b6;
}
</style>
""", unsafe_allow_html=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["fake", "real"]

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("resnet18_cat.pth", map_location=device))
    model.to(device).eval()
    return model


@st.cache_data
def load_embeddings():
    # Download the file from your dataset repo
    path = hf_hub_download(
        repo_id="aryanj10/cat-xai-embeddings",
        filename="embeddings.pkl.gz",
        repo_type="dataset"
    )

    # Load gzip-compressed pickle file
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


model = load_model()

# 🔌 Register hook for feature extraction
feature_extractor = {}
def hook_fn(module, input, output):
    feature_extractor['features'] = output.flatten(1).detach()
model.avgpool.register_forward_hook(hook_fn)

cam_extractor = GradCAM(model, target_layer="layer4")
data = load_embeddings()

# ⛩ Sidebar
st.sidebar.title("🔧 Options")
st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Try These Indices**")
st.sidebar.markdown("""
- Fake ➡ Real: `13`, `18`, `34`, `40`  
- Real ➡ Fake: `57`, `77`, `80`  
Use them to explore how the model makes errors, and how XAI helps analyze them.
""")
test_idx = st.sidebar.number_input("Select Test Index", min_value=0, max_value=len(data["test_images"])-1, value=57)
k_twins = st.sidebar.slider("Number of Twin Matches", 1, 10, value=3)

# 🎯 Load and predict
query_tensor = data["test_images"][test_idx]
true_label = data["test_labels"][test_idx]
query_input = query_tensor.unsqueeze(0).to(device)
pred_label = model(query_input).argmax(1).item()

important_cases = {
    "Real ➡ Fake": [57, 77, 80],
    "Fake ➡ Real": [13, 18, 34, 40]
}
for label, indices in important_cases.items():
    if test_idx in indices:
        st.info(f"📍 You're viewing a **{label}** misclassification test case.")


# 🔥 Grad-CAM
cam = cam_extractor(pred_label, model(query_input))[0].detach().cpu()
img_display = query_tensor.cpu() * 0.5 + 0.5
img_display = torch.clamp(img_display, 0, 1)
orig_img = to_pil_image(img_display)
cam_overlay = overlay_mask(orig_img, to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)

# 🖼 Grad-CAM visual
st.markdown("# 🐱 Fake-vs-Real Cat Classifier Explanation")

st.markdown("""
## 🧠 What This App Does

This demo uses **Explainable AI (XAI)** techniques to classify cat images as either **Real** or **Fake** (AI-generated).  
It reveals not only the **prediction**, but also the **reasoning** behind the decision using two complementary explainability tools:

### 🔥 Grad-CAM  
Shows **where** the model is "looking" by highlighting important image regions (like eyes, ears, fur).

### 🧬 Twin System  
Explains **why** a decision was made by comparing the test image's internal embedding with similar training examples.  
Think: _“This image is close to these — and the model predicted them like this.”_

---

### 🧩 Setup Summary

- **Model**: Fine-tuned `ResNet-18`
- **Classes**: `real` = 1 `fake` = 0  
- **Dataset**: 300 images (150 real + 150 fake)
- **Validation Accuracy**: 91%
""")



st.markdown("## 🎯 Grad-CAM Explanation")

st.markdown("""
Grad-CAM (**Gradient-weighted Class Activation Mapping**) visualizes **where** the model focuses when making a prediction.  
It uses backpropagation through the final convolutional layer to produce a **heatmap** of important regions.

📌 In the images below:
- **Left**: Original test image  
- **Right**: Grad-CAM overlay — bright areas = higher model attention

This helps you answer:

> _“What part of the image led the model to this decision?”_
""")
st.subheader("🔥 Grad-CAM Visualization")

cols = st.columns(2)
cols[0].image(orig_img, caption="Original Image", use_container_width=True)
cols[1].image(cam_overlay, caption=f"Grad-CAM (Pred: {CLASS_NAMES[pred_label]})", use_container_width=True)

# 🧬 Twin System
st.markdown("## 🧬 Twin System Explanation")

st.markdown("""
The **Twin System** provides an example-based explanation by retrieving **visually similar training images**.

### 🔎 How it Works:
- Embeddings from the `avgpool` layer are compared using **cosine similarity**
- The top `k` most similar training images are shown
- We display both the **true label** and **model prediction** for each

This helps you answer:

> _“What similar examples in the training set justify the model’s current prediction?”_
""")
st.subheader("🔎 Twin System Visualization")
query_emb = data["test_embeddings"][test_idx].reshape(1, -1)
distances = cosine_distances(query_emb, data["train_embeddings"])[0]
same_class_idxs = np.where(data["train_labels"] == pred_label)[0]
nearest_same = same_class_idxs[np.argsort(distances[same_class_idxs])[:k_twins]]

twin_panel = [("Query", query_tensor, true_label)] + [
    ("Twin", data["train_images"][i], data["train_labels"][i]) for i in nearest_same
]

row1, row2 = st.columns(len(twin_panel)), st.columns(len(twin_panel))
for i, (title, img_tensor, label) in enumerate(twin_panel):
    img_tensor_input = img_tensor.unsqueeze(0).to(device).requires_grad_()
    pred = model(img_tensor_input).argmax(1).item()

    cam = cam_extractor(pred, model(img_tensor_input))[0].detach().cpu()
    img_display = img_tensor.cpu() * 0.5 + 0.5
    img_display = torch.clamp(img_display, 0, 1)
    orig = to_pil_image(img_display)
    cam_overlay = overlay_mask(orig, to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)

    row1[i].image(orig, caption=f"{title}\nLabel: {CLASS_NAMES[label]}", use_container_width=True)
    row2[i].image(cam_overlay, caption=f"Grad-CAM\nPred: {CLASS_NAMES[pred]}", use_container_width=True)

cam_extractor._hooks_enabled = False

st.markdown("---")
st.markdown("### 📊 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", "91%")
    st.metric("F1 Score", "0.91")
with col2:
    st.metric("Precision (Real)", "0.94")
    st.metric("Recall (Real)", "0.88")
    st.metric("Precision (Fake)", "0.89")
    st.metric("Recall (Fake)", "0.94")


