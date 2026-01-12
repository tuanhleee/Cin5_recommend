import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# load model  exemple 
url = "https://cine5-a2ia-7.onrender.com/recommend"

payload = {
    "movie_ids": [8, 2, 3, 4, 5],
    "top_k": 5,
    "alpha": 0.2
}

try:
    res = requests.post(url, json=payload, timeout=20)
    print("Status:", res.status_code)

    try:
        print("resulat", res.json())
    except ValueError:
        print("resulat:", res.text)

except Exception as e:
    print("Error:", e)



# load images exemple 
BASE_URL = "https://res.cloudinary.com/ds84b9f8s/image/upload/v1763570455/movies_small/"

def show_image(image_id):
    """
    Télécharge et affiche une image Cloudinary en utilisant l'ID.
    """
   
    full_url = BASE_URL + image_id

    response = requests.get(full_url)
    response.raise_for_status()  

   
    img = Image.open(BytesIO(response.content))

    
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Image ID: {image_id}")
    plt.show()

show_image("img_3988.jpg")
