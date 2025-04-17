




# # app.py (Streamlit code)
# import streamlit as st
# import torch
# import clip
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import time  # <-- Import time


# def main():
#     st.title("CLIP-Based Fire Identification App for Real-Time Tracking")

#     st.write("""
#     This application leverages OpenAI's CLIP model to classify uploaded images 
#     as either has wildfire or not. By using CLIP’s pretrained 
#     capabilities, users can quickly analyze images for fire detection.

#     CLIP uses a zero-shot classification approach, comparing the uploaded image 
#     to user-provided text prompts (for example, “Wildfire Fire” vs. “No Wildfire Fire”) 
#     and returning the most likely match without requiring a separate training step.
#     """)

#     device = "cpu"  # or "cuda" if torch.cuda.is_available() else "cpu"

#     @st.cache_resource
#     def load_clip_model():
#         return clip.load("ViT-B/32", device="cpu")
    
#     model, preprocess = load_clip_model()

#     uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "jfif"])
#     if uploaded_file is not None:
#         img = Image.open(uploaded_file)
#         st.image(img, caption="Uploaded Image", use_container_width=True)
#         start_time = time.time()  # <-- Start timer

#         image = preprocess(img).unsqueeze(0).to(device)
#         text = clip.tokenize(["wildfire fire", "no wildfire fire"]).to(device)

#         with torch.no_grad():
#             image_features = model.encode_image(image)
#             text_features = model.encode_text(text)
#             logits_per_image, logits_per_text = model(image, text)
#             probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

#         end_time = time.time()  # <-- End timer
#         elapsed_time = end_time - start_time
#         st.write(f"⏱️ Inference Time: {elapsed_time:.3f} seconds")

#         labels = ["Yes", "No"]
#         fig, ax = plt.subplots(figsize=(5, 3))
#         ax.bar(labels, probs, color='gray')
#         ax.set_title("Probabilities")
#         ax.set_ylim([0, 1])
#         for i, v in enumerate(probs):
#             ax.text(i, v, f"{v:.3f}", ha='center', va='bottom')
#         st.pyplot(fig)

# if __name__ == "__main__":
#     main()








# # app.py (Streamlit code)
# import streamlit as st
# import torch
# import clip
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import time

# def main():
#     st.title("CLIP-Based Fire Identification App for Real-Time Tracking")
#     st.write("""
#     This application leverages OpenAI's CLIP model to classify uploaded images 
#     as either has wildfire or not. By using CLIP’s pretrained capabilities, 
#     users can quickly analyze images for fire detection.

#     CLIP uses a zero-shot classification approach, comparing the uploaded image 
#     to user-provided text prompts (for example, “Wildfire Fire” vs. “No Wildfire Fire”) 
#     and returning the most likely match without requiring a separate training step.
#     the model predicts a 77.5% probability that the image contains a wildfire, 
#     while assigning a 22.5% probability that it does not, 
#     indicating strong confidence in the presence of active fire in the scene.
#     """)

#     st.image("WILDFIREprob.jpg", caption="CLIP-based classification output", use_container_width=True)
    
#     device = "cpu"  # or "cuda" if torch.cuda.is_available() else "cpu"

#     @st.cache_resource
#     def load_clip_model():
#         return clip.load("ViT-B/32", device="cpu")
    
#     model, preprocess = load_clip_model()

#     uploaded_file = st.file_uploader("Upload your own image to test the model", type=["jpg", "jpeg", "png", "jfif"])
#     if uploaded_file is not None:
#         img = Image.open(uploaded_file)
#         st.image(img, caption="Uploaded image", use_container_width=True)
#         start_time = time.time()

#         image = preprocess(img).unsqueeze(0).to(device)
#         text = clip.tokenize(["wildfire fire", "no wildfire fire"]).to(device)

#         with torch.no_grad():
#             image_features = model.encode_image(image)
#             text_features = model.encode_text(text)
#             logits_per_image, logits_per_text = model(image, text)
#             probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         st.write(f"⏱️ Inference Time: {elapsed_time:.3f} seconds")

#         labels = ["Yes", "No"]
#         fig, ax = plt.subplots(figsize=(5, 3))
#         ax.bar(labels, probs, color='gray')
#         ax.set_title("Probabilities")
#         ax.set_ylim([0, 1])
#         for i, v in enumerate(probs):
#             ax.text(i, v, f"{v:.3f}", ha='center', va='bottom')
#         st.pyplot(fig)

#     st.write("---")
#     st.subheader("IoT Implimentation ")
#     st.image('camera.jpg', caption=" ")  # Replace with your actual Git image URL
    
#     st.subheader("Next Steps for IoT Deployment")
#     st.write("""
#     The next step is to bring this AI-powered fire detection model into the real world by building a small, 
#     smart device using a Raspberry Pi and a video camera. This device will be able to monitor the landscape 
#     in real time, looking for signs of wildfire and analyzing the video feed on the spot using the CLIP model.

#     By adding GPS and connecting the device to other sensors (like temperature or smoke detectors), 
#     we can create a system that not only detects possible fire activity but also knows where it's happening. 
#     This combination of artificial intelligence, geolocation, and sensor data lays the foundation for a smart, 
#     spatially aware monitoring tool.

#     While the initial focus is wildfire detection, this kind of system can be expanded to help in many 
#     other areas such as tracking invasive plant species, monitoring infrastructure in remote areas, 
#     or even supporting land and utility management. Ultimately, it's about building a flexible, 
#     intelligent sensing platform that helps people better understand and respond to what is 
#     happening on the landscape in real time.
#     """)

# if __name__ == "__main__":
#     main()






# app.py (Streamlit code)
import streamlit as st
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    st.title("CLIP Project One \n CLIP-Based Fire Identification for Real-Time Tracking")
    st.write("""
    This application leverages OpenAI's CLIP model to classify uploaded images 
    as either has wildfire or not. By using CLIP’s pretrained capabilities, 
    users can quickly analyze images for fire detection.

    CLIP uses a zero-shot classification approach, comparing the uploaded image 
    to user-provided text prompts (for example, “Wildfire Fire” vs. “No Wildfire Fire”) 
    and returning the most likely match without requiring a separate training step.
    
    The model predicts a 77.5% probability that the image contains a wildfire, 
    while assigning a 22.5% probability that it does not, 
    indicating strong confidence in the presence of active fire in the scene.
    """)

    st.image("WILDFIREprob.jpg", caption="CLIP-based classification output", use_container_width=True)

    device = "cpu"  # or "cuda" if torch.cuda.is_available() else "cpu"

    @st.cache_resource
    def load_clip_model():
        return clip.load("ViT-B/32", device="cpu")
    
    model, preprocess = load_clip_model()

    uploaded_file = st.file_uploader("Upload your own image to test the model", type=["jpg", "jpeg", "png", "jfif"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded image", use_container_width=True)
        start_time = time.time()

        image = preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize(["wildfire fire", "no wildfire fire"]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"⏱️ Inference Time: {elapsed_time:.3f} seconds")

        labels = ["Yes", "No"]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(labels, probs, color='gray')
        ax.set_title("Probabilities")
        ax.set_ylim([0, 1])
        for i, v in enumerate(probs):
            ax.text(i, v, f"{v:.3f}", ha='center', va='bottom')
        st.pyplot(fig)

        # IoT section appears only after model runs
        st.write("---")
        st.subheader("IoT Implementation")
        st.image('camera.jpg', caption="Example camera setup for landscape monitoring", use_container_width=True)

        st.subheader("Next Steps for IoT Deployment")
        st.write("""
        The next step is to bring this AI-powered fire detection model into the real world by building a small, 
        smart device using a Raspberry Pi and a video camera. This device will be able to monitor the landscape 
        in real time, looking for signs of wildfire and analyzing the video feed on the spot using the CLIP model.

        By adding GPS and connecting the device to other sensors (like temperature or smoke detectors), 
        we can create a system that not only detects possible fire activity but also knows where it's happening. 
        This combination of artificial intelligence, geolocation, and sensor data lays the foundation for a smart, 
        spatially aware monitoring tool.

        While the initial focus is wildfire detection, this kind of system can be expanded to help in many 
        other areas such as tracking invasive plant species, monitoring infrastructure in remote areas, 
        or even supporting land and utility management. Ultimately, it's about building a flexible, 
        intelligent sensing platform that helps people better understand and respond to what is 
        happening on the landscape in real time.
        """)

if __name__ == "__main__":
    main()



