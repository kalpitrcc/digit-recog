import os
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import logging
import io

# Add logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = f"{BASE_DIR}/model"

def load_model():
    model = tf.keras.models.load_model(MODEL_DIR)
    return model

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):
    try:
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents)).convert('L').resize((28, 28))
        image_array = np.array(image)
        image_array = image_array.reshape(1, 28, 28) / 255.0

        model = load_model()
        predictions = model.predict(image_array)
        predicted_label = np.argmax(predictions[0])

        return {"predicted_label": int(predicted_label), "class_name": class_names[predicted_label]}
    except Exception as e:
        logger.error(f"Error processing the image: {e}")
        raise HTTPException(status_code=500, detail="Error processing the image")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
