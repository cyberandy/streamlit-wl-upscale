import streamlit as st
import requests
from inference_client import Client
from jina import DocumentArray, Document
from PIL import Image
from io import BytesIO
import os


def get_file_size(image_bytes):
    return len(image_bytes) / 1024  # Convert to KB


def get_image_size(image):
    width, height = image.size
    return width, height


def upscale_image(uri, model, scale):
    doc = Document(
        uri=uri, tags={"image_format": "png", "output_path": "upscaled_image.png"}
    )
    docs = DocumentArray([doc])

    result = model.upscale(docs=docs, scale=scale)
    return result


def main():
    st.sidebar.image("assets/logo-wordlift.png", width=200)

    st.sidebar.title("Image Upscaling Settings")
    access_token = st.sidebar.text_input("Access Token")
    model_name = st.sidebar.text_input("Model Name", value="LapSRN_x2")
    width = st.sidebar.number_input("Image Width", value=1200)
    height = st.sidebar.number_input("Image Height", value=-1)

    st.title("Image Upscaling App")

    image_uri = st.text_input("Image URL")

    if st.button("Upscale"):
        if not access_token or not model_name or not image_uri:
            st.warning("Please enter Access Token, Model Name, and Image URL")
        else:
            st.info("Upscaling in progress...")
            client = Client(token=access_token)
            model = client.get_model(model_name)
            result = upscale_image(image_uri, model, f"{width}:{height}")

            original_image = Image.open(BytesIO(requests.get(image_uri).content))
            original_weight = get_file_size(requests.get(image_uri).content)
            original_width, original_height = get_image_size(original_image)

            for r in result:
                upscaled_image_bytes = r.blob
                upscaled_image = Image.open(BytesIO(upscaled_image_bytes))

                # Preserve aspect ratio while resizing
                upscaled_image.thumbnail((width, height), Image.ANTIALIAS)

                upscaled_weight = get_file_size(upscaled_image_bytes)
                upscaled_width, upscaled_height = get_image_size(upscaled_image)

                st.image(
                    upscaled_image, caption="Upscaled Image", use_column_width=True
                )
                st.write("Original Weight: {:.2f} KB".format(original_weight))
                st.write("Upscaled Weight: {:.2f} KB".format(upscaled_weight))
                st.write(
                    "Original Size: {} x {}".format(original_width, original_height)
                )
                st.write(
                    "Upscaled Size: {} x {}".format(upscaled_width, upscaled_height)
                )

                image_bytes = BytesIO()
                upscaled_image.save(image_bytes, format="PNG")
                st.download_button(
                    "Download", image_bytes.getvalue(), file_name="upscaled_image.png"
                )


if __name__ == "__main__":
    main()
