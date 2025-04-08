import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')  # Convert from bytes to string

# Example usage
image_path = "face_20250401_161145_0.jpg"  # Change this to your image path
base64_str = image_to_base64(image_path)
print(base64_str)
