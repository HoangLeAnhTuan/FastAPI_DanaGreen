import base64
import httpx
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Dense
import keras.applications.xception as xception
from keras.models import load_model
import qrcode
from io import BytesIO
from fastapi.responses import StreamingResponse
from Cryptodome.Cipher import AES
from Cryptodome.Util.Padding import pad, unpad
from Cryptodome.Random import get_random_bytes
from pydantic import BaseModel

# FastAPI app instance
app = FastAPI()

# Set up templates directory for HTML rendering
templates = Jinja2Templates(directory="templates")

# Constants
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320
IMAGE_CHANNELS = 3
categories = {
    0: 'metal', #battery
    1: 'paper', #cardboard
    2: 'plastic', #glass
    3: 'metal',
    4: 'paper',
    5: 'plastic'
}


# In-memory trash history
trashHistory = {}

# Build model
def build_model():
    xception_layer = xception.Xception(include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), weights='imagenet')
    xception_layer.trainable = False

    model = Sequential()
    model.add(tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

    def xception_preprocessing(img):
        return xception.preprocess_input(img)

    model.add(Lambda(xception_preprocessing))
    model.add(xception_layer)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Dense(len(categories), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model

model = build_model()
model.load_weights('model/model.weights.h5')

# Predict route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = await file.read()
        np_arr = np.frombuffer(image, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error decoding image")

    if frame is None:
        raise HTTPException(status_code=400, detail="Error decoding image")

    # Process image
    frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    frame_array = np.expand_dims(frame_resized, axis=0)

    # Make prediction
    preds = model.predict(frame_array)
    pred_class = preds.argmax(1)
    pred_class_name = categories[pred_class[0]]

    # Get confidence score
    confidence = np.max(preds)
    # Check confidence threshold
    if confidence <= 0.4:
        return JSONResponse({
            'message': 'Low acc',
            'confidence': round(float(confidence),3)
        })
    else:
        # Update trash history if confidence is greater than 0.4
        if pred_class_name in trashHistory:
            trashHistory[pred_class_name] += 1
        else:
            trashHistory[pred_class_name] = 1
        
        return JSONResponse({
            'prediction': pred_class_name,
            'confidence': round(float(confidence),3)
        })

# Key and IV for AES
key = b'danagreenkey0807'  # Example key 16 bytes
iv = b'danagreenivv1605'  # Example IV 16 bytes for AES CBC

# Encrypt data using AES CBC
def encrypt_data(data):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(data.encode('utf-8'), AES.block_size))
    return base64.b64encode(ciphertext).decode('utf-8')

# Decrypt data from ciphertext (base64)
def decrypt_data(ciphertext_base64):
    ciphertext = base64.b64decode(ciphertext_base64)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ciphertext), AES.block_size).decode('utf-8')

# QR code route
@app.get("/qrcode")
async def get_qrcode():
    if not trashHistory:
        raise HTTPException(status_code=404, detail="No trash history to display")

    category_abbreviations = {
        "metal": "Me",
        "plastic": "Pl",
        "paper": "pa",
    }

    trash_data = ""
    for key, value in trashHistory.items():
        abbreviation = category_abbreviations.get(key, key[0].upper())
        trash_data += f"{abbreviation}{value}"
    encrypted_trash_data = encrypt_data(trash_data)

    img = qrcode.make(encrypted_trash_data)
    buf = BytesIO()
    img.save(buf)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

class DecryptRequest(BaseModel):
    qrcode_text: str

# Route to decrypt QR code from ciphertext
@app.post("/decrypt_qrcode")
async def decrypt_qr_code(request: DecryptRequest):
    try:
        decrypted_data = decrypt_data(request.qrcode_text)
        return {"decrypted_data": decrypted_data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Route to clear trash history
@app.get("/end")
async def end_trash_history():
    trashHistory.clear()
    return JSONResponse({"message": "Trash history cleared successfully"})

# HTML Form to Upload Image and Display Result
@app.get("/test", response_class=HTMLResponse)
async def get_test_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/test", response_class=HTMLResponse)
async def handle_test(request: Request, file: UploadFile = File(...)):
    try:
        image = await file.read()
        np_arr = np.frombuffer(image, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error decoding image")

    if frame is None:
        raise HTTPException(status_code=400, detail="Error decoding image")

    frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    frame_array = np.expand_dims(frame_resized, axis=0)

    preds = model.predict(frame_array)
    pred_class = preds.argmax(1)
    pred_class_name = categories[pred_class[0]]
    confidence = np.max(preds)

    if confidence > 0.4:
        if pred_class_name in trashHistory:
            trashHistory[pred_class_name] += 1
        else:
            trashHistory[pred_class_name] = 1

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "prediction": pred_class_name,
        "confidence": f"{confidence:.2f}"
    })

# Predict with camera route
@app.get("/predict_with_camera")
async def predict_with_camera():
    # Initialize camera
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Camera not found")

    # Capture an image
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to capture image")

    # Process image
    frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    _, buffer = cv2.imencode('.jpg', frame_resized)
    image_data = buffer.tobytes()

    # Send image to /predict
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:5000/predict/", files={"file": ("image.jpg", image_data, "image/jpeg")})

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to get prediction")

    return response.json()

# Start the server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
