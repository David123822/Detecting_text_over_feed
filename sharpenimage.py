import tkinter as tk  # Import the Tkinter library for creating GUI applications
import PIL.Image  # Import the Image class from PIL (Pillow) for image processing
import PIL.ImageTk  # Import the ImageTk class to convert PIL images to Tkinter images
import cv2  # Import OpenCV for computer vision tasks
import numpy as np  # Import NumPy for numerical operations, including image array manipulation

def sharpen_image(image, intensity, base_value):
    """
    Function to sharpen a given image using a sharpening kernel.
    Args:
    - image: The input image in PIL format.
    - intensity: The intensity value from the slider (Scale widget).
    - base_value: The base sharpening value from the Spinbox.

    Returns:
    - The sharpened image as a PIL image.
    """
    # Convert the PIL image to a NumPy array for OpenCV processing
    image_np = np.array(image)

    area_to_enhance = base_value + intensity

    # Define a sharpening kernel, adjusting the center value based on the base value + intensity
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  area_to_enhance, -1],
                               [-1, -1, -1]])

    # Apply the sharpening kernel to the image using OpenCV's filter2D function
    sharpened_np = cv2.filter2D(image_np, -1, sharpen_kernel)

    # Convert the NumPy array back to a PIL image
    sharpened_pil = PIL.Image.fromarray(cv2.cvtColor(sharpened_np, cv2.COLOR_BGR2RGB))

    return sharpened_pil

def update_image(intensity):
    """
    Function to update the displayed image based on the sharpening intensity and base value.
    Args:
    - intensity: The sharpening intensity from the slider.
    """
    # Get the base value from the Spinbox
    base_value = int(base_value_spinbox.get())

    # Sharpen the image based on the current slider value and base value
    sharpened_img = sharpen_image(original_image, intensity, base_value)

    # Convert the sharpened image to ImageTk format for display in Tkinter
    img_tk = PIL.ImageTk.PhotoImage(sharpened_img)

    # Update the label with the new image
    image_label.imgtk = img_tk  # Keep a reference to avoid garbage collection
    image_label.configure(image=img_tk)

# Load the original image using PIL
image_path = 'image.jpg'  # Replace with your actual image path
original_image = PIL.Image.open(image_path)

# Set up the main Tkinter window
root = tk.Tk()
root.geometry("800x600")
root.title("Image Sharpening with Slider and Base Value Counter")

# Label to display the image
image_label = tk.Label(root)
image_label.pack()

# Label for the Spinbox
spinbox_label = tk.Label(root, text="Base Sharpening Value")
spinbox_label.pack()

# Spinbox for adjusting the base sharpening value
base_value_spinbox = tk.Spinbox(root, from_=1, to=20, width=5)
base_value_spinbox.pack()

# Slider (Scale widget) for adjusting the sharpening intensity
slider = tk.Scale(root, from_=0, to=10, orient=tk.HORIZONTAL, label="Sharpening Intensity",
                  command=lambda value: update_image(int(value)))
slider.set(5)  # Set default value for the slider
slider.pack()

# Display the initial image
update_image(slider.get())

# Start the Tkinter main loop
root.mainloop()