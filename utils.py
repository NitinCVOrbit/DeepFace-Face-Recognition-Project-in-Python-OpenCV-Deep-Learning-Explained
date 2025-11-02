import cv2
import os
from deepface import DeepFace


# -------------------- FACE DETECTION --------------------
def face_detection(img_path, original_image):

    # Load input image
    img = cv2.imread(img_path)
    # Resize for fixed layout
    img = cv2.resize(img, (381, 420))

    # Place input image in the left panel of template
    original_image[191:611, 433:814] = img

    # Copy image for drawing detections
    img1 = img.copy()
    
    # Detect faces in the image
    results = DeepFace.extract_faces(img1, enforce_detection=False)

    # Draw bounding boxes around detected faces
    for face in results:
        region = face['facial_area']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Place detected face image in the right panel of template
    original_image[191:611, 873:1254] = img1

    return original_image


# -------------------- FACE SIMILARITY --------------------
def face_similarity(img1_path, img2_path, original_image, u_pressed):

    if u_pressed:
        # Load second image and place in right box
        img2 = cv2.imread(img2_path)
        img2 = cv2.resize(img2, (370, 352))
        original_image[209:561, 890:1260] = img2   

        # Show paths of both images
        print(img1_path)
        print(img2_path) 

        # Compare both faces with DeepFace
        result = DeepFace.verify(img1_path, img2_path)
        distance = result["distance"]

        # Convert distance into similarity %
        similarity_percentage = (1 - distance) * 100

        # Display similarity percentage on template
        cv2.putText(original_image,f'{int(similarity_percentage)}%',
                    (913,645), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255,0,0), 2)

        # Reset flag
        u_pressed = False

    else:    
        # First image selected → place in left box
        u_pressed = True
        img = cv2.imread(img2_path)
        img = cv2.resize(img, (370, 352))
        original_image[209:561, 464:834] = img

    return original_image, u_pressed


# -------------------- FACE FIND --------------------
def find_face(img_path, original_image):   

    # Load and resize input image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (380, 419))

    # Place image in template
    original_image[171:590, 584:964] = img

    # Database folder path
    db_path = "Dataset_Images"

    # Search for matching faces in database
    results = DeepFace.find(img_path=img_path, db_path=db_path, enforce_detection=False)

    # If matches are found
    if len(results) > 0 and not results[0].empty:

        # Print top matches with distance
        print(results[0][["identity", "distance"]])  

        # Create results folder if not exists
        if not os.path.exists('Find_DB'):
            os.mkdir('Find_DB')  

        filename = os.path.basename(img_path)

        # Create subfolder for current image
        if not os.path.exists(f'Find_DB/{filename}'):
            os.mkdir(f'Find_DB/{filename}') 
        
        counter = 0

        # Iterate over found matches
        for path, distance in zip(results[0]['identity'], results[0]['distance']):

            # Convert distance into similarity %
            similarity_percentage = (1 - distance) * 100    # 0.9 - 1 = 0.1 * 100 = 10

            # Save matches with >50% similarity
            if similarity_percentage > 50:
                img_find = cv2.imread(path)
                basename = os.path.basename(path)
                cv2.imwrite(f'Find_DB/{filename}/{basename}', img_find)
                counter += 1
                
    else:
        # No matches found
        print("No matching face found.")
        counter = 0
        filename = os.path.basename(img_path)

    # Show number of matches
    cv2.putText(original_image, f'{counter}', (765,642), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)    

    # Show result folder path or no match
    if counter > 0:
        cv2.putText(original_image, f'Find_DB/{filename}', (765,685), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    else:    
        cv2.putText(original_image, f'No match found', (765,685), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    return original_image


# -------------------- FACE INFO --------------------
def face_info(img_path, original_image):

    # Load and resize input image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (380, 419))
    original_image[192:611, 433:813] = img

    # Analyze attributes: age, gender, race, emotion
    result = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'])

    print(result)

    # Extract attributes
    age = result[0]['age']
    gender = result[0]['dominant_gender']
    emotion = result[0]['dominant_emotion']
    race = result[0]['dominant_race']

    # Display results on template
    cv2.putText(original_image, f'{emotion}', (1064,298), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)        
    cv2.putText(original_image, f'{age}', (1064,360), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv2.putText(original_image, f'{gender}', (1064,422), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)        
    cv2.putText(original_image, f'{race}', (1064,484), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    return original_image
