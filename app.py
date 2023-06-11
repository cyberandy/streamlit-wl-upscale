import streamlit as st
import requests
from inference_client import Client
from jina import DocumentArray, Document
from PIL import Image
from io import BytesIO
import os
import numpy as np
from streamlit_image_comparison import image_comparison


## Streamlit Settings ##

st.set_page_config(
    page_title="Image Upscaling and Captioning 🖼️",
    page_icon="assets/fav-ico.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://wordlift.io/book-a-demo/",
        "About": "This is an *demo* created by @cyberandy and the WordLift team! Lern more: https://wordlift.io/blog/en/image-seo-using-ai/",
    },
)

## Helper Functions ##


def get_file_size(image_bytes):
    return len(image_bytes.getvalue()) / 1024  # Convert to KB


def get_image_size(image):
    width, height = image.size
    return width, height


def upscale_image(uri, model, scale):
    doc = Document(
        uri=uri, tags={"image_format": "png", "output_path": "upscaled_image.jpg"}
    )
    docs = DocumentArray([doc])
    result = model.upscale(docs=docs, scale=scale)
    return result


def generate_caption(client, image):
    caption = client.caption(image=image)
    return caption


## Main App ##


def main():
    st.sidebar.image("assets/logo-wordlift.png", width=200)
    st.sidebar.title("Image Upscaling and Captioning Settings")
    access_token = st.sidebar.text_input("Access Token")
    upscale_model_name = st.sidebar.text_input("Upscale Model Name", value="LapSRN_x2")
    caption_model_name = st.sidebar.text_input(
        "Caption Model Name", value="Salesforce/blip2-flan-t5-xl"
    )
    width = st.sidebar.number_input("Image Width", value=1200)
    height = st.sidebar.number_input("Image Height", value=-1)

    st.title("Image Upscaling and Captioning 🖼️")

    image_uri = st.text_input("Image URL")

    if access_token and upscale_model_name and image_uri and st.button("Upscale"):
        try:
            st.info("Upscaling in progress...")
            client = Client(token=access_token)
            upscale_model = client.get_model(upscale_model_name)
            result = upscale_image(image_uri, upscale_model, f"{width}:{height}")

            # ... (rest of the code inside the if block) ...

        except ConnectionError:
            st.error(
                "Connection error. Please check your internet connection and try again."
            )

    if (
        access_token
        and caption_model_name
        and image_uri
        and st.button("Generate Caption")
    ):
        try:
            st.info("Generating caption...")
            client = Client(token=access_token)
            caption_model = client.get_model(caption_model_name)
            image_content = requests.get(image_uri).content
            caption = generate_caption(caption_model, image_content)
            st.write("Caption: {}".format(caption))

        except ConnectionError:
            st.error(
                "Connection error. Please check your internet connection and try again."
            )


## Run App ##


## Run App ##

if __name__ == "__main__":
    main()
