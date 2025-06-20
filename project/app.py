import streamlit as st
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI
import google.generativeai as genai

# Get your Google API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the generative models
# For image analysis (multimodal)
vision_model = genai.GenerativeModel('gemini-pro-vision')

# For text-only analysis, use a modern Gemini text model.
# 'gemini-1.5-flash-latest' is generally recommended for speed and cost-efficiency.
# 'gemini-1.5-pro-latest' for more complex reasoning.
text_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Changed from 'gemini-pro'

def get_nutritional_info_from_image(image_data, prompt):
    """
    Analyzes an image with a given prompt using Gemini Pro Vision
    to extract nutritional information.
    """
    try:
        response = vision_model.generate_content([prompt, image_data])
        return response.text
    except Exception as e:
        # Catch potential ResourceExhausted or other API errors
        return f"Error analyzing image: {e}. Please try again later or check your API quota."

def get_nutritional_info_from_text(text_input, prompt_template):
    """
    Analyzes text input with a given prompt using the text-optimized Gemini model
    to extract nutritional information.
    """
    try:
        full_prompt = f"{prompt_template}\n\nFood Item: {text_input}"
        response = text_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        # Catch potential ResourceExhausted or other API errors
        return f"Error analyzing text: {e}. Please try again later or check your API quota."

# Streamlit UI
st.set_page_config(page_title="🥗 NutriAI - Instant Nutritional Information", layout="centered")

st.header("🥗 NutriAI - Instant Nutritional Information")
st.write("Upload a food image or describe a food item to get its nutritional breakdown!")

# Input method selection
input_method = st.radio("Choose input method:", ("Upload Image", "Enter Text Description"))

nutritional_prompt_base = """
You are NutriAI, a highly accurate and concise AI assistant specialized in providing nutritional information.
For the given food item (either from an an image or a text description), provide the following details:
- **Food Item Identified:**
- **Approximate Serving Size:** (e.g., 100g, 1 cup, 1 piece)
- **Estimated Calories:**
- **Macronutrient Breakdown (in grams):**
    - Protein:
    - Carbohydrates:
    - Fats:
- **Key Micronutrients (mention 2-3 significant ones if applicable, e.g., Vitamin C, Iron):**
- **Brief Healthy Eating Tip (related to this food if possible):**

If you cannot identify the food, state that clearly: "Cannot identify the food item. Please try again with a clearer input."
Be brief and to the point. Do not add conversational filler.
Format the output using markdown for readability.
"""

output_placeholder = st.empty() # Placeholder for displaying results

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.info("Analyzing image... This may take a moment.")

        # Convert image to bytes for Gemini API
        img_byte_arr = io.BytesIO()
        # Save as JPEG for potentially smaller file size and common compatibility
        image.save(img_byte_arr, format='JPEG') 
        img_byte_arr = img_byte_arr.getvalue()

        # Prepare image data for Gemini API
        image_data = {
            'mime_type': 'image/jpeg', 
            'data': img_byte_arr
        }

        with st.spinner("Getting nutritional info from image..."):
            response_text = get_nutritional_info_from_image(image_data, nutritional_prompt_base)
            output_placeholder.markdown(f"**Nutritional Information:**\n{response_text}")

elif input_method == "Enter Text Description":
    text_input = st.text_area("Describe the food item:", height=100, placeholder="e.g., 'a bowl of oatmeal with blueberries', '1 medium apple'")

    if st.button("Get Nutritional Info"):
        if text_input.strip(): # Check if text input is not empty or just whitespace
            st.info("Analyzing text description... This may take a moment.")
            with st.spinner("Getting nutritional info from text..."):
                response_text = get_nutritional_info_from_text(text_input, nutritional_prompt_base)
                output_placeholder.markdown(f"**Nutritional Information:**\n{response_text}")
        else:
            st.warning("Please enter a food description.")

st.markdown("---")
st.markdown("Powered by Google Gemini Generative AI")