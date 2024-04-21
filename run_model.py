import os
import onnxruntime
import numpy as np
from PIL import Image

# Load the ONNX model
onnx_model = onnxruntime.InferenceSession("") # direct model path
img_height, img_width = 150, 150

# Choose some images from the test directory
# Change to the desired subfolder
test_images_dir = '' # img folder path
test_image_files = os.listdir(test_images_dir)[:5]  # Choose first 5 images

# Make predictions for each image
for image_file in test_image_files:
    img_path = os.path.join(test_images_dir, image_file)
    img = Image.open(img_path).resize((img_height, img_width))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Run inference
    input_name = onnx_model.get_inputs()[0].name
    output_name = onnx_model.get_outputs()[0].name
    prediction = onnx_model.run(
        [output_name], {input_name: img_array.astype(np.float32)})[0]

    # Print the prediction array for debugging
    # print("Prediction Array:", prediction)

    # Get class labels (assuming you already have these defined)
    # Define your class labels here
    class_labels = ['blight', 'common_rust', 'gray_leaf_spot', 'healthy']

    # Print the predicted class label
    predicted_class_index = np.argmax(prediction)
    if predicted_class_index < len(class_labels):
        predicted_class = class_labels[predicted_class_index]
        print(f"Image: {image_file}, Predicted Class: {predicted_class}")
    else:
        print(
            f"Image: {image_file}, Predicted Class Index: {predicted_class_index} is out of range.")
