# -------------------- IMPORTS --------------------
import cv2
import tkinter as tk
from tkinter import filedialog
from utils import face_detection, face_similarity, find_face, face_info


# -------------------- SCREEN IMAGE PATHS --------------------
# Mapping between screen numbers (1–4) and their placeholder images
screen_images = {
    '1': 'Resources/face_detection.png',
    '2': 'Resources/face_similarity.png',
    '3': 'Resources/find_face.png',
    '4': 'Resources/face_info.png'
}


# -------------------- INITIAL STATE --------------------
current_screen = '1'  # Start with screen 1
original_image = cv2.imread(screen_images['1'])  # Load default image
cv2.imshow("DeepFace", original_image)  # Show it in OpenCV window

# Variables for screen 2 (similarity check)
screen_2_img1_path = None
screen_2_u_pressed = False  


# -------------------- MAIN FEATURE HANDLER --------------------
def deepface_feature(img_path):
    """
    Decide which DeepFace function to call
    based on the current screen (1–4).
    """
    global screen_2_img1_path, screen_2_u_pressed, original_image

    if current_screen == '1':      # Face detection
        return face_detection(img_path, original_image)

    elif current_screen == '2':    # Face similarity
        original_image, screen_2_u_pressed = face_similarity(
            screen_2_img1_path, img_path, original_image, screen_2_u_pressed
        )
        return original_image

    elif current_screen == '3':    # Find face in dataset
        return find_face(img_path, original_image)

    elif current_screen == '4':    # Face attributes (age, gender, etc.)
        return face_info(img_path, original_image)


# -------------------- SHOW SCREEN --------------------
def show_screen(screen_key):
    """Show the placeholder image for the chosen screen (1–4)."""
    global original_image
    img = cv2.imread(screen_images[screen_key])
    cv2.imshow("DeepFace", img)
    original_image = img


# -------------------- FILE SELECTOR --------------------
def select_file():
    """
    Open a file dialog to select an image,
    then process it according to the active screen.
    """
    global screen_2_img1_path, screen_2_u_pressed, original_image

    # Open dialog
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
    )

    # If no file is chosen
    if not file_path:
        print("No file selected.")
        return original_image

    # For screen 2, store the first selected image
    if current_screen == '2' and not screen_2_u_pressed:   # 2 and true
        screen_2_img1_path = file_path

    # Run DeepFace feature
    original_image = deepface_feature(file_path)

    # Update window
    cv2.imshow("DeepFace", original_image)
    cv2.imwrite(f"{file_path}_{current_screen}.png",original_image)
    # return original_image 


# -------------------- KEYBOARD LOOP --------------------
while True:
    key = cv2.waitKey(0)  # Wait for key press

    # Change screen (1–4)
    if chr(key) in ['1', '2', '3', '4']:
        current_screen = chr(key)
        show_screen(current_screen)

    # Press U → select image
    elif key in [ord('u'), ord('U')]:
        if current_screen in screen_images:
            select_file()  
        else:
            print("Press 1–4 before using U.")

    # Press Q → second step for screen 2
    elif key in [ord('q'), ord('Q')]:
        print("U Pressed:", screen_2_u_pressed)
        if current_screen == '2':
            if screen_2_u_pressed:
                select_file()
            else:
                print("Press U first.")
        else:
            print("Press 1–4 before using Q.")

    # Press 5 or ESC → exit program
    elif key in [ord('5'), 27]:
        break

    # Wrong key
    else:
        print("Invalid key. Use 1–4 to select a screen.")


# -------------------- CLEANUP --------------------
cv2.destroyAllWindows()

