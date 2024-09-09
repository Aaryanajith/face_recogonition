import cv2
import os
from imgbeddings import imgbeddings
import numpy as np
from PIL import Image
import csv
from sklearn.metrics.pairwise import cosine_similarity


# alg = "/home/interstellar/face_recogonition/haarcascade_frontalface_default.xml"
# haar_cascade = cv2.CascadeClassifier(alg)
# file_name = "/home/interstellar/face_recogonition/picture.jpg"
# img = cv2.imread(file_name, 0)

# gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# faces = haar_cascade.detectMultiScale(
#     gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100)
# )

# i = 0

# for x, y, w, h in faces:

#     cropped_image = img[y : y + h, x : x + w]

#     target_file_name = 'stored-faces/' + str(i) + '.jpg'
#     cv2.imwrite(
#         target_file_name,
#         cropped_image,
#     )
#     i = i + 1

ibed = imgbeddings()

with open('face_embeddings.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'Embedding']) 

    for filename in os.listdir("stored-faces"):
        img = Image.open(os.path.join("stored-faces", filename))
        embedding = ibed.to_embeddings(img)
        
        embedding_flattened = embedding.flatten().tolist()
        
        writer.writerow([filename] + embedding_flattened)

inp_image = Image.open("/home/interstellar/face_recogonition/tony.jpg")
inp_image_embedding = ibed.to_embeddings(inp_image)

best_match_filename = None
best_match_score = -1

with open('face_embeddings.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader) 

    for row in reader:
        filename = row[0]  
        embedding_str = row[1:]

        stored_embedding = np.array([float(x) for x in embedding_str]).reshape(1, -1)
        
        current_embedding_reshaped = inp_image_embedding.reshape(1, -1)

        similarity_score = cosine_similarity(current_embedding_reshaped, stored_embedding)[0][0]


        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match_filename = filename

if best_match_filename:
    print(f"The closest match is: {best_match_filename} with a similarity score of {best_match_score}")
else:
    print("No match found.")