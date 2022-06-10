import streamlit as st
from tensorflow import keras
from PIL import Image

model = keras.models.load_model('modele_nounou')


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None
        
    st.button('Predict')
    
    
def predict(img):
    pdc = model.predict(img)
    st.write(pdc)
        

    
def main():
    st.title('Image upload demo')

    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(image)


if __name__ == '__main__':
    main()
    
    


