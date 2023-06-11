import streamlit as st
import requests
from inference_client import Client
from jina import DocumentArray, Document
from PIL import Image
from io import BytesIO
import os
import numpy as np
from streamlit_image_comparison import image_comparison


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
        st.info("Upscaling in progress...")
        client = Client(token=access_token)
        upscale_model = client.get_model(upscale_model_name)
        result = upscale_image(image_uri, upscale_model, f"{width}:{height}")

        original_image = Image.open(BytesIO(requests.get(image_uri).content))
        original_weight = get_file_size(BytesIO(requests.get(image_uri).content))
        original_width, original_height = get_image_size(original_image)

        for r in result:
            upscaled_image_bytes = r.blob
            upscaled_image = Image.open(BytesIO(upscaled_image_bytes))

            # Preserve aspect ratio while resizing
            upscaled_image.thumbnail((width, height), Image.ANTIALIAS)

            # Save the upscaled image in memory in JPEG format for calculating the compressed weight
            image_bytes = BytesIO()
            upscaled_image.save(image_bytes, format="JPEG", quality=80)
            upscaled_weight = get_file_size(image_bytes)

            upscaled_width, upscaled_height = get_image_size(upscaled_image)

            # Comparing Original Image and Upscaled Image
            image_comparison(
                original_image,
                upscaled_image,
                label1="Original Image",
                label2="Upscaled Image",
            )

            st.write("Original Image Weight: {:.2f} KB".format(original_weight))
            st.write(
                "Upscaled (compressed) Image Weight: {:.2f} KB".format(upscaled_weight)
            )
            st.write(
                "Original Image Size: {} x {}".format(original_width, original_height)
            )
            st.write(
                "Upscaled Image Size: {} x {}".format(upscaled_width, upscaled_height)
            )

            st.download_button(
                "Download", image_bytes.getvalue(), file_name="upscaled_image.jpg"
            )

    if (
        access_token
        and caption_model_name
        and image_uri
        and st.button("Generate Caption")
    ):
        st.info("Generating caption...")
        client = Client(token=access_token)
        caption_model = client.get_model(caption_model_name)
        image_content = requests.get(image_uri).content
        caption = generate_caption(caption_model, image_content)
        st.write("Caption: {}".format(caption))


if __name__ == "__main__":
    main()
