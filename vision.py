import google.generativeai as genai
import os

def identify_product_from_image(image_bytes, api_key):
    """
    Uses Gemini 1.5 Flash to identify the product name from an image.
    """
    try:
        genai.configure(api_key=api_key)
        prompt = "Identify the smartphone or electronic device model from this image. Return ONLY the core device model name (e.g., 'Apple iPhone 15 Pro', 'Samsung Galaxy S24 Ultra'). IMPORTANT: Ignore any cases, covers, or accessories. Even if the phone has a case on it, strictly return the name of the PHONE itself. Do not include words like 'case', 'cover', or 'box'."
        image_parts = [{"mime_type": "image/jpeg", "data": image_bytes}]
        
        # Try multiple models to ensure one works
        models_to_try = ['gemini-flash-latest', 'gemini-2.5-flash', 'gemini-2.0-flash']
        last_error = None
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, image_parts[0]])
                return response.text.strip()
            except Exception as e:
                last_error = str(e)
                continue
                
        return f"Error: All models failed. Last error: {last_error}"
    except Exception as e:
        return f"Error: {str(e)}"
