import tkinter as tk  # Import the Tkinter library for creating GUI applications
import PIL.Image  # Import the Image class from PIL (Pillow) for image processing
import PIL.ImageTk  # Import the ImageTk class to convert PIL images to Tkinter images
import cv2  # Import OpenCV for computer vision tasks
import numpy as np  # Import NumPy for numerical operations, including image array manipulation
import pymsgbox  # Import pymsgbox for pop-up messages
import pyperclip
from langdetect import detect, DetectorFactory
import pytesseract

DetectorFactory.seed = 0

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

cap = None

def cam_feed(label):
    """
    Function to start the camera feed and continuously update the image on the Tkinter label.
    """
    global cap

    # Stop the previous feed update loop by releasing the camera if it's already running
    if cap is not None:
        cap.release()

    # Create a new VideoCapture object based on the selected camera source
    cap = cv2.VideoCapture(camera_source.get())

    # Setting the camera properties before capturing frames
    cap.set(3, 450)  # Set the width of the video capture to 450 pixels
    cap.set(4, 450)  # Set the height of the video capture to 450 pixels
    #cap.set(5, 120)  # Set the frame rate of the video capture to 120 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))  # Set the codec to MJPG

    def update_frame():
        """
        Inner function to update the camera feed frame by frame.
        """
        ret, frame = cap.read()  # Capture a frame from the camera
        #frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror effect)

        if ret:
            # Convert the BGR frame to RGB for display in Tkinter
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the RGB frame to a PIL image
            img = PIL.Image.fromarray(rgb)

            # Decode barcodes over the feed and update the image with any detected barcodes
            img = detect_text_over_feed(img)

            # Convert the PIL image to an ImageTk object for display in Tkinter
            imgTk = PIL.ImageTk.PhotoImage(img)
            label.imgtk = imgTk  # Keep a reference to avoid garbage collection
            label.configure(image=imgTk)  # Update the label to display the new image

        # Schedule the update_frame function to be called again after 10 milliseconds
        label.after(10, update_frame)

    # Start the update loop
    update_frame()

def take_picture():
    """
    Function to capture a single image from the camera feed and save it as a file.
    """
    global cap

    if cap is None or not cap.isOpened():
        pymsgbox.alert('Please start the video feed first.', 'Warning')  # Alert user if the camera feed is not running
    else:
        ret, frame = cap.read()  # Capture a frame from the camera
        #frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror effect)

        if ret:
            # Save the captured frame as a JPEG file
            cv2.imwrite('image.jpg', frame)
            print("Image captured and saved as 'image.jpg'")
        else:
            pymsgbox.alert('Failed to capture image.', 'Error')


def sharpen_image(image):
    """
    Function to sharpen a given image using a sharpening kernel.
    Args:
    - image: The input image in PIL format.
    
    Returns:
    - The sharpened image as a PIL image.
    """
    # Convert the PIL image to a NumPy array for OpenCV processing
    image_np = np.array(image)

    # Define a sharpening kernel
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

    # Apply the sharpening kernel to the image using OpenCV's filter2D function
    sharpened_np = cv2.filter2D(image_np, -1, sharpen_kernel)

    # Convert the NumPy array back to a PIL image
    sharpened_pil = PIL.Image.fromarray(cv2.cvtColor(sharpened_np, cv2.COLOR_BGR2RGB))

    return sharpened_pil


def detect_text(image_path="image.jpg"):
    """
    Detects text from the provided image, sharpens it, and updates the display.
    """
    # Clear the text widget to remove any previously displayed text
    text_widget.delete('1.0', tk.END)

    # Release the camera if it is still running to avoid conflicts with file handling
    if cap is not None:
        cap.release()

    try:
        # Open the image from the provided path using PIL (Pillow)
        img = PIL.Image.open(image_path)

        # Sharpen the image
        img = sharpen_image(img)

        # Convert the PIL image to a NumPy array for OpenCV operations
        img_np = np.array(img)

        if img_np.size == 0:
            raise ValueError("The image is empty or corrupted.")
        
    except (FileNotFoundError, ValueError) as e:
        # Display an error message to the user in case the image is not found or corrupted
        pymsgbox.alert(f"Error: {str(e)}")
        return

    # Convert to grayscale for better text recognition
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Use Tesseract's image_to_data function to extract text and their corresponding bounding box information.
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # If no text is detected, inform the user by adding a message to the text widget
    if len(data['text']) == 0:
        text_widget.insert(tk.END, "No text detected")
        return

    prev_top = -1  # Variable to track the 'top' position of the previous word's bounding box

    # Iterate through each detected text string in the 'data' dictionary
    for i, text in enumerate(data['text']):
        if text.strip():  # Ignore empty or whitespace-only strings
            # Extract the bounding box coordinates and dimensions for the current text entry
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

            # Draw a rectangle around the detected text in the image using OpenCV
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # If the current 'top' value is different from the previous one, it means we are in a new line
            if prev_top != -1 and abs(y - prev_top) > 10:  # Adjust threshold as needed
                text_widget.insert(tk.END, "\n")  # Add a new line in the text widget

            # Insert the detected text into the text_widget to display it in the order it appears in the image
            text_widget.insert(tk.END, f"{text} ")

            # Update the previous 'top' value to the current one for line grouping
            prev_top = y

    # Convert the NumPy array back to a PIL image for displaying in Tkinter
    img_pil = PIL.Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    # Convert the PIL image to an ImageTk object to be displayed in Tkinter
    img_tk = PIL.ImageTk.PhotoImage(image=img_pil)

    # Update the label with the new image that includes the drawn bounding boxes
    cam_label.imgtk = img_tk  # Keep a reference to avoid garbage collection
    cam_label.configure(image=img_tk)  # Update the label with the new image





def detect_text_over_feed(image):
    """
    Function to detect text over the camera feed, draw bounding boxes around detected text,
    and update the image in the Tkinter label.
    """
    # Convert the PIL image to a NumPy array for OpenCV processing
    image_np = np.array(image)

    # Convert to grayscale for better text detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Use Tesseract to extract text and bounding box information
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # If no text is detected, display a message in the text widget
    if len(data['text']) == 0:
        text_widget.delete('1.0', tk.END)  # Clear the previous text
        text_widget.insert(tk.END, "No text detected")
        return image  # Return the original image without modification

    # Clear the text widget before inserting new text
    text_widget.delete('1.0', tk.END)

    prev_top = -1  # To track the line's vertical position for grouping text by line

    # Iterate through each detected text string in the Tesseract output
    for i, text in enumerate(data['text']):
        if text.strip():  # Ignore empty or whitespace-only strings
            # Get the bounding box coordinates and dimensions for the current text block
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

            # Draw a red rectangle around the detected text
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Insert the text into the text widget in a way that preserves line breaks
            if prev_top != -1 and abs(y - prev_top) > 10:
                text_widget.insert(tk.END, "\n")  # Start a new line in the widget

            # Insert the text into the text widget
            text_widget.insert(tk.END, f"{text} ")

            # Update the top position for the next iteration
            prev_top = y

    # Convert the updated NumPy array back to a PIL image for displaying in Tkinter
    img_pil = PIL.Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    return img_pil  # Return the processed image with bounding boxes drawn



# Main Tkinter application window
root = tk.Tk()
root.geometry("800x500")  # Set the window size
root.title("Barcode detection")

camera_source = tk.IntVar(value=0)  # Default to 0 (built-in camera)

# Frame for displaying the camera feed
cam_frame = tk.Frame(root)
cam_frame.place(x=10, y=10, width=450, height=450)

# Label to display the camera feed
cam_label = tk.Label(cam_frame, text="Display")
cam_label.pack()

# Button to capture a picture from the camera feed
take_picture_button = tk.Button(root, text="Take a Picture", command=take_picture, bg="blue", fg="white")
take_picture_button.place(x=470, y=10, width=120, height=40)

# Button to detect barcodes from a saved image
detect_code_button = tk.Button(root, text="Detect Text", command=lambda: detect_text(), bg="green", fg="white")
detect_code_button.place(x=470, y=60, width=120, height=40)

# Button to start the camera feed
show_feed = tk.Button(root, text="Start feed", command=lambda: cam_feed(cam_label), bg="blue", fg="white")
show_feed.place(x=470, y=110, width=120, height=40)

# Button to exit the application
exit_button = tk.Button(root, text="Exit", command=root.destroy, bg="red", fg="white")
exit_button.place(x=470, y=160, width=120, height=40)

# Radio buttons to select the camera source (built-in or external webcam)
def_cam = tk.Radiobutton(root, text="Built-in Camera (opens faster)", variable=camera_source, value=0)
def_cam.place(x=470, y=210)
web_cam = tk.Radiobutton(root, text="Webcam (opens slower)", variable=camera_source, value=1)
web_cam.place(x=470, y=240)

# Treeview to display the detected barcodes
text_widget = tk.Text(root)
text_widget.place(x=470, y=270, width=320, height=225)

# Start the Tkinter event loop
root.mainloop()

# Release the capture and close windows when the program ends
if cap is not None:
    cap.release()

cv2.destroyAllWindows  # Clean up OpenCV resources