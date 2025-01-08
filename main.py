import random

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from tensorflow.keras.utils import load_img
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
import os

random.seed(a=None, version=2)

set_verbosity(INFO)


def get_mean_std_per_batch(image_dir, df, H=320, W=320):
    sample_data = []
    for img in df.sample(100)["Image"].values:
        image_path = os.path.join(image_dir, img)
        sample_data.append(
            np.array(load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data, axis=(0, 1, 2, 3))
    std = np.std(sample_data, axis=(0, 1, 2, 3), ddof=1)
    return mean, std


def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    mean, std = get_mean_std_per_batch(image_dir, df, H=H, W=W)
    img_path = os.path.join(image_dir, img)
    x = load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x
def load_image_normalize(img, image_dir, mean, std, H=320, W=320):
    path = os.path.join(image_dir, img)
    x = load_img(path, target_size=(H, W))
    x -= mean
    x /= std
    x = np.expand_dims(x, axis=0)
    return x


# def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
#     """GradCAM method for visualizing input saliency."""
#     y_c = input_model.output[0, cls]
#     conv_output = input_model.get_layer(layer_name).output
#     grads = K.gradients(y_c, conv_output)[0]
   
#     gradient_function = K.function([input_model.input], [conv_output, grads])

#     output, grads_val = gradient_function([image])
#     output, grads_val = output[0, :], grads_val[0, :, :, :]

#     weights = np.mean(grads_val, axis=(0, 1))
#     cam = np.dot(output, weights)

#     # Process CAM
#     cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
#     cam = np.maximum(cam, 0)
#     cam = cam / cam.max()
#     return cam

def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
    """GradCAM method for visualizing input saliency."""
    with tf.GradientTape() as tape:
        # cast the image tensor to a float-32 data type, while preserving existing values if possible
        inputs = tf.cast(image, tf.float32) 

        # Watch the input tensor during gradient calculations
        tape.watch(inputs) 

        # Obtain predictions from the model for the given input
        preds = input_model(inputs)

        # Get the prediction score for the target class
        y_c = preds[0, cls]

    # Calculate gradients of the target class score with respect to the model's input
    # (This captures how changes in the input affect the prediction for that class)
    grads = tape.gradient(y_c, inputs)[0] 

    # Obtain the output of the specified layer
    # (Typically a convolutional layer's output is used for GradCAM)
    conv_output = input_model.get_layer(layer_name).output

    # Obtain the output of the specified layer for the given input image
    # (This gives us the activations of that layer for this specific image)
    iterate = tf.keras.models.Model([input_model.input], [conv_output, input_model.output])
    conv_output, predictions = iterate(inputs)

    # Average the gradients over the width and height dimensions (spatial dimensions)
    # (This provides a class-discriminative localization map)
    pooled_grads = K.mean(grads, axis=(0, 1, 2)) 

    # Iterate through the filters/channels in the convolutional layer's output
    for i in range(conv_output.shape[-1]):
        # Multiply each filter's activations by the corresponding weighted gradient
        # (This weights the activations based on their importance for the target class)
        conv_output[:, :, :, i] *= pooled_grads[i]

    # Average the weighted activations across the channels/filters
    # (This produces a single heatmap highlighting the regions most important for the class)
    heatmap = K.mean(conv_output, axis=-1)

    # Apply ReLU to the heatmap to keep only positive activations
    heatmap = np.maximum(heatmap[0], 0)

    # Normalize the heatmap to a range of 0-1
    heatmap /= np.max(heatmap) 
    heatmap = cv2.resize(heatmap, (W, H)) # Resize heatmap to match image dimensions
    heatmap = np.uint8(255 * heatmap) # Convert heatmap values to 8-bit integers (0-255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply colormap to visualize heatmap

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(image[0], 0.5, heatmap, 0.5, 0)
    return superimposed_img

    
def compute_gradcam(model, img, image_dir, df, labels, selected_labels,
                    layer_name='bn'):
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model.predict(preprocessed_input)

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img, image_dir, df, preprocess=False), cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            plt.subplot(151 + j)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            plt.imshow(load_image(img, image_dir, df, preprocess=False),
                       cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1


def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals
