import streamlit as st
import requests
from inference_client import Client
from PIL import Image
from io import BytesIO
from streamlit_image_comparison import image_comparison


## Streamlit Settings ##

st.set_page_config(
    page_title="Image Upscaling and Captioning 🖼️",
    page_icon="assets/fav-ico.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://wordlift.io/book-a-demo/",
        "About": "This is an *demo* created by @cyberandy and the WordLift team! Learn more: https://wordlift.io/blog/en/image-seo-using-ai/",
    },
)


## Helper Functions ##


def get_file_size(image_bytes):
    return len(image_bytes.getvalue()) / 1024  # Convert to KB


def get_image_size(image):
    width, height = image.size
    return width, height


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
    width = st.sidebar.number_input("Image Width", value=500)
    height = st.sidebar.number_input("Image Height", value=-1)

    st.title("Image Upscaling and Captioning 🖼️")

    image_uri = st.text_input("Image URL")

    if access_token and upscale_model_name and image_uri and st.button("Upscale"):
        st.info("Upscaling in progress...")
        client = Client(token=access_token)
        model = client.get_model(upscale_model_name)
        image = requests.get(image_uri).content

        # Using updated upscale function that handles compression
        result = model.upscale(
            image=image,
            scale=f"{width}:{height}",
            output_path="upscaled_image.jpeg",
            quality=80,
        )
        upscaled_image = Image.open(BytesIO(result))

        original_image = Image.open(BytesIO(image))
        original_weight = get_file_size(BytesIO(image))
        original_width, original_height = get_image_size(original_image)

        upscaled_weight = get_file_size(BytesIO(result))
        upscaled_width, upscaled_height = get_image_size(upscaled_image)

        # Comparing Original Image and Upscaled Image
        image_comparison(
            original_image,
            upscaled_image,
            label1="Original",
            label2="Upscaled",
        )

        st.write("Original Image Weight: {:.2f} KB".format(original_weight))
        st.write(
            "Upscaled (compressed) Image Weight: {:.2f} KB".format(upscaled_weight)
        )
        st.write("Original Image Size: {} x {}".format(original_width, original_height))
        st.write("Upscaled Image Size: {} x {}".format(upscaled_width, upscaled_height))

        st.download_button("Download", result, file_name="upscaled_image.jpeg")

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


## Run App ##

if __name__ == "__main__":
    main()
