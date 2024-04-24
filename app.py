import numpy as np
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

logo_img = Image.open("skin_logo.jpg")
page_config = {"page_title": "Skin Lesion Classification", "page_icon": logo_img, "layout": "centered"}
st.set_page_config(**page_config)

page = option_menu(
    menu_title=None,
    options=["Classification", "Code"],
    icons=["motherboard", "file-earmark-code"],
    default_index=0,
    orientation="horizontal",
    styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "black", "font-size": "14px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "red"}
            }
)

# Prediction page
if page == "Classification":

    st.text("")
    st.markdown("""***This Machine Learning application lets you classify dermatological images of pigmented lesions
                   into 7 categories.***""")
    st.text("")
    file = st.file_uploader('***Upload your image***')

    def getPrediction(input_image):
        classes = ['Actinic keratoses', 'Basal cell carcinoma',
                   'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma',
                   'Melanocytic nevi', 'Vascular lesions']
        le = LabelEncoder()
        le.fit(classes)
        le.inverse_transform([2])

        # Load model
        my_model = load_model("HAM10000_100epochs.h5")

        # Resize to same size as training images
        SIZE = 32
        img = np.asarray(Image.open(input_image).resize((SIZE, SIZE)))

        # Scale pixel values
        img = img / 255.

        # Get it ready as input to the network
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = my_model.predict(img)

        # Convert prediction to class name
        pred_class = le.inverse_transform([np.argmax(pred)])[0]
        return pred_class

    if st.button('Predict'):
        output = getPrediction(file)
        st.write(output)
        op = Image.open(file)
        st.image(op)

# Code page
if page == "Code":

    st.text("")
    st.write("###### If you are more interested in the code you can directly jump into these repositories :")
    st.text("")
    st.caption("**DEPLOYMENT** : ***[link](https://github.com/sangoleshubham20/HAM-streamlit)***")
    st.caption("**MODELLING** : ***[link](https://github.com/sangoleshubham20/HAM-training)***")
