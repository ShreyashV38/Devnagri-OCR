"""
app.py - Devanagari Character Recognizer (Gradio UI)

Usage: python app.py
"""

import sys
import os

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.keras.models import load_model
import gradio as gr

import cv2

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'devanagari_master.h5')
LABELS_PATH = os.path.join(PROJECT_ROOT, 'backend', 'label_classes.npy')

IMG_SIZE = 32

# ---------------------------------------------------------------------------
# LOAD MODEL AND LABELS AT STARTUP
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print(f'\nERROR: Model not found at {MODEL_PATH}')
    print('Run train_model.py first:  python train_model.py\n')
    sys.exit(1)

if not os.path.exists(LABELS_PATH):
    print(f'\nERROR: Labels not found at {LABELS_PATH}')
    print('Run train_model.py first:  python train_model.py\n')
    sys.exit(1)

print('Loading model...')
model = load_model(MODEL_PATH)
label_classes = np.load(LABELS_PATH, allow_pickle=True)
num_classes = len(label_classes)
print(f'Model loaded: {num_classes} classes')
print(f'Classes: {list(label_classes)}')


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
def predict(image):
    if image is None:
        return "No image provided.", ""

    # To grayscale
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            if np.min(alpha) < 255:
                image = alpha
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Invert if light background
    bg = np.median(np.concatenate([
        image[0,:], image[-1,:], image[:,0], image[:,-1]
    ]))
    if bg > 127:
        image = 255 - image

    # OTSU threshold - better than fixed 50
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological close to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Find largest contour - ignore noise dots
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Keep only large contours (>1% of image area)
        h_img, w_img = image.shape
        min_area = h_img * w_img * 0.01
        big = [c for c in contours if cv2.contourArea(c) > min_area]
        if big:
            # Merge all big contours bbox
            xs = [cv2.boundingRect(c)[0] for c in big]
            ys = [cv2.boundingRect(c)[1] for c in big]
            x2s = [cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2] for c in big]
            y2s = [cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3] for c in big]
            x,y,x2,y2 = min(xs),min(ys),max(x2s),max(y2s)
            image = image[y:y2, x:x2]

    # Square + margin
    h, w = image.shape
    side = max(h, w)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    margin = int(side * 0.15)
    image = cv2.copyMakeBorder(
        image,
        pad_y+margin, side-h-pad_y+margin,
        pad_x+margin, side-w-pad_x+margin,
        cv2.BORDER_CONSTANT, value=0
    )

    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    cv2.imwrite("debug_input.png", img)

    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    probs = model.predict(img, verbose=0)[0]

    top_idx = np.argmax(probs)
    top_label = label_classes[top_idx]
    top_conf = probs[top_idx] * 100
    top_prediction = f"{top_label}  ({top_conf:.1f}%)"

    top5_indices = np.argsort(probs)[::-1][:5]
    lines = []
    for rank, idx in enumerate(top5_indices, 1):
        lines.append(f"{rank}. {label_classes[idx]:<20} -- {probs[idx]*100:.1f}%")
    top5_text = "\n".join(lines)

    return top_prediction, top5_text
# ---------------------------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Devanagari Character Recognizer") as app:
    gr.Markdown("# Devanagari Character Recognizer")
    gr.Markdown("Upload an image of a single Devanagari character.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload Character Image")
            recognize_btn = gr.Button("Recognize", variant="primary")

        with gr.Column():
            top_output = gr.Textbox(label="Predicted Character", interactive=False)
            top5_output = gr.Textbox(label="Top 5 Predictions", lines=6, interactive=False)

    recognize_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[top_output, top5_output]
    )

    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=[top_output, top5_output]
    )


if __name__ == '__main__':
    print('\nLaunching Gradio app...')
    app.launch(share=False, server_name="127.0.0.1")
