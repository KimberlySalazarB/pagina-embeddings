# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import openai
import pickle
from openai import OpenAI
#from openai import ChatCompletion
from PIL import Image
from io import BytesIO
import subprocess
from openai import OpenAI
from sklearn.exceptions import InconsistentVersionWarning
import warnings



# Función para obtener el contenido de un archivo desde una URL
def obtener_contenido_archivo(url):
    try:
        respuesta = requests.get(url)
        respuesta.raise_for_status()  # Lanza una excepción si hay un error en la solicitud
        return respuesta.content
    except requests.exceptions.RequestException as e:
        print("Error al obtener el archivo:", e)
        return None

#modelo
def modelo():
    url_archivo = "https://raw.githubusercontent.com/KimberlySalazarB/pagina-embeddings/main/modelo_clasificacion.pkl"
    contenido = obtener_contenido_archivo(url_archivo)
    if contenido is not None:
        return contenido
    else:
        return None
    
def parse_embeddings(embedding_str):
    try:
        # Eliminar corchetes y dividir por comas para obtener los números
        values = embedding_str.strip('[]').split(', ')
        # Convertir los valores a números flotantes y devolver como lista
        return [float(val) for val in values]
    except:
        # Si no se puede convertir, devuelve una lista vacía
        return []
def obtener_incrustaciones(data,column_name,api_key):
        # Seleccionar modelo  "gpt-3.5-turbo"
    client = OpenAI(api_key=api_key)
    #model = "gpt-3.5-turbo"
    # Itera a través de la columna y obtén las incrustaciones para cada texto.
    embeddings = []
    for texto in data[column_name]:
        embedding = client.embeddings.create(input=texto, model="text-embedding-ada-002").data[0].embedding
        embeddings.append(embedding)

    data['Embeddings'] = embeddings
    data['Embeddings'] =data['Embeddings'].apply(parse_embeddings)

    # Obtener la longitud máxima de las incrustaciones
    max_length = max(len(embedding) for embedding in data['Embeddings'])

    # Aplicar padding a las incrustaciones para que todas tengan la misma longitud
    nuevos_padded_embeddings = [embedding + [0.0] * (max_length - len(embedding)) for embedding in data['Embeddings']]

    X_nuevos = np.array(nuevos_padded_embeddings)
    df_resultados = pd.DataFrame(X_nuevos)

    return df_resultados


# Función principal
def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Bienvenidos a la página! ❤️")

    st.markdown(
        """
        Aquí, nos sumergimos en conversaciones significativas relacionadas con la vacuna contra el 
        Virus del Papiloma Humano (VPH). Utilizamos un clasificador especializado para analizar y 
        categorizar los comentarios de manera precisa y eficiente.
        El objetivo principal es dar los comentarios antivacunas para entender las diversas perspectivas
        expresadas por la comunidad en torno a la vacuna contra el VPH.  

        Nuestro clasificador asigna números específicos a cada comentario 
        para reflejar la postura del autor. La clasificación es la siguiente:

        0: Postura contraria a la vacuna contra el VPH (Antivacuna).  
        1: Postura a favor de la vacuna contra el VPH (Provacuna).  
        2: Expresa dudas relacionadas con la vacuna contra el VPH.  
        3: Comentarios que no se relacionan con la vacuna contra el VPH.  
    """
    )

    # Botón para ocultar/mostrar la API de OpenAI
    api_key = st.text_input("API Key de OpenAI", type="password")
    # Mostrar advertencia si no se ha ingresado la API Key
    if not api_key:
        st.warning("Ingrese su API Key de OpenAI.")
        return

    column_name = st.text_input("Ingrese el nombre de la columna que contiene los comentarios:")
    
    st.write("Versión de scikit-learn:", sklearn.__version__)
                      
    uploaded_file = st.file_uploader("Cargar archivo", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            file_ext = uploaded_file.name.split(".")[-1]
            if file_ext == "csv":
                data = pd.read_csv(uploaded_file)
            elif file_ext == "xlsx":
                data = pd.read_excel(uploaded_file)
            
            st.write("Datos cargados:")
            st.write(data)
            
            # Clasificar los comentarios si se ha proporcionado la API Key
            if api_key:
                #openaiapi_key="'"+ str(api_key) + "'"
                X_nuevos = obtener_incrustaciones(data, column_name, api_key)
                st.write(X_nuevos)
                # Añadir ceros adicionales para igualar el número de características esperado por el modelo
                X_nuevos_con_padding = np.pad(X_nuevos, ((0, 0), (0, 22)), mode='constant')
                modelo_cargado = pickle.loads(modelo())
                # Hacer predicciones con el modelo cargado utilizando los datos con padding
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                predicciones_nuevas = modelo_cargado.predict(X_nuevos_con_padding)
                if 'Clasificación_gpt_4' not in data.columns:
                    data['Clasificación_gpt_4'] = ''
                for index, row in data.iterrows():
                    data.at[index, 'Clasificación_gpt_4'] = predicciones_nuevas
                    st.write(data)

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

    # Mostrar la imagen desde una URL
    url_imagen = "https://raw.githubusercontent.com/KimberlySalazarB/paginaprueba/main/Imagen3.jpg"
    contenido_imagen = obtener_contenido_archivo(url_imagen)
    if contenido_imagen is not None:
        imagen = Image.open(BytesIO(contenido_imagen))
        st.image(imagen, caption='Imagen desde la URL')

    # Mostrar comentarios antivacunas al hacer clic en un botón
    if st.button("Mostrar comentarios antivacunas"):
        comentarios_antivacunas = data[data['Clasificación_gpt_4'] == 0][column_name].tolist()
        st.subheader("Comentarios antivacunas encontrados:")
        if comentarios_antivacunas:
            for comentario in comentarios_antivacunas:
                st.write(comentario)
        else:
            st.write("No se encontraron comentarios antivacunas.")

    # Mostrar comentarios antivacunas al hacer clic en un botón
    if st.button("Mostrar comentarios dudas"):
        comentarios_antivacunas = data[data['Clasificación_gpt_4'] == 2][column_name].tolist()
        st.subheader("Comentarios de dudas:")
        if comentarios_antivacunas:
            for comentario in comentarios_antivacunas:
                st.write(comentario)
        else:
            st.write("No se encontraron comentarios dudas.")
                    
if __name__ == "__main__":
    run()