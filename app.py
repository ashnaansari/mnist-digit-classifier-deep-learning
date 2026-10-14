
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------
# 1. Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Classifier")
st.write("Draw a digit from 0 to 9 and let your AI predict it!")

# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "notebooks" / "mnist_model.npz"

try:
    with np.load(MODEL_PATH) as model:
        W1 = model["W1"]
        b1 = model["b1"]
        W2 = model["W2"]
        b2 = model["b2"]

except FileNotFoundError:
    st.error(f"Model not found: {MODEL_PATH}")
    st.stop()

# --------------------------------------------------
# 3. Drawing canvas
# --------------------------------------------------

st.subheader("Draw your digit below")

st.caption(
    "Draw one large digit in the center. "
    "Use a continuous stroke and avoid touching the edges."
)

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="digit_canvas"
)

# --------------------------------------------------
# 4. Prepare the drawing for MNIST
# --------------------------------------------------

def preprocess_image(canvas_data):
    # Convert the canvas data to an 8-bit RGBA image
    rgba = Image.fromarray(
        np.asarray(canvas_data).astype(np.uint8)
    ).convert("RGBA")

    # Composite transparency onto a solid black background
    black_background = Image.new(
        "RGBA", rgba.size, (0, 0, 0, 255)
    )
    rgba = Image.alpha_composite(black_background, rgba)

    # Convert to grayscale
    gray = rgba.convert("L")

    # Find the area containing the white digit
    gray_array = np.array(gray)
    mask = gray_array > 20

    if not np.any(mask):
        return None

    ys, xs = np.where(mask)

    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1

    # Crop around the digit
    digit = gray.crop((left, top, right, bottom))

    # Add padding to make the crop square
    width, height = digit.size
    side = max(width, height)

    square = Image.new("L", (side, side), 0)

    x_offset = (side - width) // 2
    y_offset = (side - height) // 2

    square.paste(digit, (x_offset, y_offset))

    # Resize the digit to 20 x 20 pixels
    # MNIST digits typically occupy approximately this area
    digit = square.resize(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Place it in the center of a 28 x 28 black image
    final_image = Image.new("L", (28, 28), 0)
    final_image.paste(digit, (4, 4))

    return final_image


# --------------------------------------------------
# 5. Predict the digit
# --------------------------------------------------

if st.button("Predict Digit", type="primary"):

    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")

    else:
        processed_image = preprocess_image(
            canvas_result.image_data
        )

        if processed_image is None:
            st.warning("No digit detected. Please draw a digit first.")

        else:
            # Normalize pixels from 0-255 to 0-1
            input_image = (
                np.array(processed_image, dtype=np.float32)
                / 255.0
            )

            # Flatten 28 x 28 pixels into 784 inputs
            input_image = input_image.reshape(1, 784)

            # Forward pass through the trained NumPy network
            hidden_input = input_image @ W1 + b1
            hidden_output = np.maximum(0, hidden_input)
            output = hidden_output @ W2 + b2

            # Softmax: convert scores into probabilities
            shifted_output = (
                output - np.max(output, axis=1, keepdims=True)
            )
            exp_output = np.exp(shifted_output)

            probabilities = exp_output / np.sum(
                exp_output, axis=1, keepdims=True
            )

            # Find the predicted digit
            predicted_digit = int(
                np.argmax(probabilities[0])
            )

            confidence = float(
                probabilities[0, predicted_digit] * 100
            )

            # Display prediction
            st.success(f"Predicted Digit: {predicted_digit}")

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

            # Display the actual image sent to the model
            st.subheader("Image Given to the Model")

            st.image(
                processed_image,
                caption="28 × 28 grayscale image",
                width=200
            )

            # Display probabilities for all ten digits
            st.subheader("Probabilities for Each Digit")

            probability_values = probabilities[0] * 100

            probability_table = {
                "Digit": [str(i) for i in range(10)],
                "Probability (%)": [
                    round(float(p), 2)
                    for p in probability_values
                ]
            }

            st.bar_chart(
                probability_table,
                x="Digit",
                y="Probability (%)"
            )
