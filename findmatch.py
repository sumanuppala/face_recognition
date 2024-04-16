import face_recognition

image_of_gill = face_recognition.load_image_file('facerecog/pics/group/gill.jpg')
gill_face_encoding = face_recognition.face_encodings(image_of_gill)[0]

unknown_image = face_recognition.load_image_file('facerecog/pics/unknown/mahesh.jpg')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces(
    [gill_face_encoding], unknown_face_encoding)

if results[0]:
    print('This is gill')
else:
    print('This is NOT gill')