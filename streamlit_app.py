import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import numpy as np

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

recyclable_lille = ["plastic","glass","metal","paper","cardboard"]

villeneuve_papier = ['papier','cardboard']

villeneuve_plastique = ['plastic','glass','metal']

villeneuve_organique = ['organic']


def load_model():
    model = keras.models.load_model('modele_nounou')
    return model

def choice():
    option = st.selectbox(
        'Dans quelle ville résidez vous ?',
        ('Lille', "Villeneuve d'Ascq"))

    st.write('You selected:', option)
    if option== 'Lille':
        image_lille = Image.open('lille.jpg')
        st.image(image_lille)

    if option== "Villeneuve d'Ascq":
        image_villeneuve = Image.open('villeneuve_d_ascq.png')
        st.image(image_villeneuve)
    return option

def mode():
    mode= st.radio("Comment voulez vous utiliser l'application ?",("Prendre une photo","Utiliser une photo"))
    return mode 


def take_image():
    picture = st.camera_input("Take a picture")
    if picture:
        picture = Image.open(picture)
        picture = picture.resize((500, 500))
        image_data = picture.getvalue()
        
        st.image(image_data)
        return io.BytesIO(image_data)
    else:
        return None
        
def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file:
        uploaded_file = Image.open(uploaded_file)
        uploaded_file = uploaded_file.resize((500,500))
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return io.BytesIO(image_data)
    else:
        return None


def predict(model, image,option):
    img = keras.preprocessing.image.load_img(image, target_size=(256, 256))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pdc = model.predict(img_array)
    # st.write(pdc[0]*100 , "\n", classes)
    if option == 'Lille':
        if classes[np.argmax(pdc)] in recyclable_lille:
            st.write("recyclable")
            st.write(classes[np.argmax(pdc)])

        else:
            st.write("non recyclable")
            st.write(classes[np.argmax(pdc)])

    if option =="Villeneuve d'Ascq":
        if classes[np.argmax(pdc)] in villeneuve_papier:
            st.write("Compartiment papier")
            st.write(classes[np.argmax(pdc)])

        elif classes[np.argmax(pdc)] in villeneuve_plastique:
            st.write("Compartiment plastique / conserve / verre")
            st.write(classes[np.argmax(pdc)])
        elif classes[np.argmax(pdc)] in villeneuve_organique:
            st.write("Compartiment déchets organiques")
            st.write(classes[np.argmax(pdc)])
        else:
            st.write("Compartiment non recyclable")
            st.write(classes[np.argmax(pdc)])

    # st.write("Prediction: ", classes[np.argmax(pdc)], pdc[0][np.argmax(pdc)])
        

    
def main():
    st.title('Image upload demo')
    model = load_model()
    option_1 = choice()
    
    photo_mode = mode()

    if photo_mode == "Prendre une photo":
        image = take_image()

    elif photo_mode == "Utiliser une photo":
        image = load_image()
    result = st.button('Run on image')
    if result:
        # st.write('Calculating results...')
        predict(model, image,option_1)


if __name__ == '__main__':
    main()
    
    

