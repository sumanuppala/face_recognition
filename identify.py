import face_recognition
from PIL import Image, ImageDraw
from PIL import ImageFont

# Load the default font
font = ImageFont.load_default()

image_of_bill = face_recognition.load_image_file('facerecog/pics/known/Bill Gates.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]

image_of_steve = face_recognition.load_image_file('facerecog/pics/known/Steve Jobs.jpg')
steve_face_encoding = face_recognition.face_encodings(image_of_steve)[0]

image_of_ElonMusk = face_recognition.load_image_file('facerecog/pics/known/Elon Musk.jpg')
ElonMusk_face_encoding = face_recognition.face_encodings(image_of_ElonMusk)[0]

#  Create arrays of encodings and names
known_face_encodings = [
  bill_face_encoding,
  steve_face_encoding,
  ElonMusk_face_encoding
]

known_face_names = [
  "bill.JPG",
  "steve.jpg",
  "ElonMusk.jpg"
]

# Load test image to find faces in
test_image = face_recognition.load_image_file('facerecog/pics/group/bill-steve-elon.jpg')

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown Person"

    # If match
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    # Draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0))

    # Draw label
    text_box = draw.textbbox((left, bottom, right, bottom + 20), name, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]

    draw.rectangle(((left, bottom), (right, bottom + text_height + 10)), fill=(255, 255, 0), outline=(255, 255, 0))
    draw.text((left + 6, bottom + 5), name, fill=(0, 0, 0), font=font)


del draw

# Display image
pil_image.show()

# Save image
#pil_image.save('identify.jpg')
