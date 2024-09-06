import cv2

alg = "/home/interstellar/face_recogonition/.haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

file_name = "/home/interstellar/face_recogonition/picture.jpg"

img = cv2.imread(file_name, 0)

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100)
)


i = 0

for x, y, w, h in faces:

    cropped_image = img[y : y + h, x : x + w]

    target_file_name = 'stored-faces/' + str(i) + '.jpg'
    cv2.imwrite(
        target_file_name,
        cropped_image,
    )
    i = i + 1