import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

def load_model():
    model = keras.models.load_model('modele_nounou')
    return model


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return keras.preprocessing.image.load_img(image_data, target_size=(256, 256))
    else:
        return None
        
    st.button('Predict')
    
    
def predict(model, img):
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pdc = model.predict(img_array)
    st.write(pdc)
        

    
def main():
    st.title('Image upload demo')
    model = load_model()
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, image)


if __name__ == '__main__':
    main()
    
    


