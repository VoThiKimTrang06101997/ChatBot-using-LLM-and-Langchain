import base64
import openai
from utils import load_config

config = load_config()

# Configure OpenAI API key
openai.api_key = 'openai-api-key'

def convert_bytes_to_base64(image_bytes):
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    return "data:image/jpeg;base64," + encoded_string


def handle_image(image_bytes, user_input):
    # Set the OpenAI API key correctly from the configuration
    openai.api_key = openai.api_key  

    # Convert the image to base64 to possibly use it within HTML or store it, but not sending to OpenAI
    image_base64 = convert_bytes_to_base64(image_bytes)
    
    # Since OpenAI API cannot directly process images, handle the user's text input about the image
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an AI trained to provide information about images."},
                {"role": "user", "content": user_input}  # User's message about the image
            ],
            max_tokens=2000,
            temperature=0.5
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return str(e)
    
    