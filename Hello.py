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
import yaml
from openai import OpenAI
#from openai import ChatCompletion
from PIL import Image
from io import BytesIO
import subprocess
from openai import OpenAI
import sklearn
from sklearn.exceptions import InconsistentVersionWarning
import warnings
from pandasai import SmartDataframe
#from pandasai.llm import OpenAI as pandaAI


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
    url_archivo = "https://raw.githubusercontent.com/KimberlySalazarB/pagina-embeddings/main/modelo_clasificacionpickle.pkl"
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
def obtener_incrustaciones(data, column_name, api_key):
    try:
        # Verificar la API Key
        if not api_key:
            st.write("Error: API Key de OpenAI no proporcionada.")
            return None

        # Seleccionar modelo
        client = OpenAI(api_key=api_key)
        
        # Verificar la estructura de los datos
        if column_name not in data.columns:
            st.write("Error: La columna especificada no existe en los datos.")
            return None

        # Iterar a través de la columna y obtener las incrustaciones para cada texto
        embeddings = []
        for texto in data[column_name]:
            try:
                embedding = client.embeddings.create(input=texto, model="text-embedding-ada-002").data[0].embedding
                embeddings.append(embedding)
            except Exception as e:
                print("Error al obtener incrustaciones para el texto:", texto)
                print("Excepción:", e)

        if not embeddings:
            print("Advertencia: No se pudieron obtener incrustaciones para ningún texto.")
            return None

        # Obtener la longitud máxima de las incrustaciones
        max_length = 2000

        # Aplicar padding a las incrustaciones para que todas tengan la misma longitud
        nuevos_padded_embeddings = []
        for embedding in embeddings:
            padding_length = max_length - len(embedding) #2000
            padded_embedding = embedding + [0.0] * padding_length
            nuevos_padded_embeddings.append(padded_embedding)

        X_nuevos = np.array(nuevos_padded_embeddings)
        

        return X_nuevos
    
    except Exception as e:
        st.write("Error general al obtener incrustaciones:", e)
        return None


# Función principal

# Función para descargar DataFrame como archivo CSV
def download_csv(data):
    csv = data.to_csv(index=False)
    return csv
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
        categorizar los comentarios con una presición en clasificar comentarios antivacunas de 0.917 
        y comentarios de dudas de 0.9.

        
        El objetivo principal de categorizar y analizar los comentarios recibidos en las publicaciones, 
        con el fin de identificar las dudas y preocupaciones del público objetivo, así como las 
        opiniones antivacunas. 

        Nuestro clasificador asigna números específicos a cada comentario 
        para reflejar la postura del autor. La clasificación es la siguiente:

        0: Postura contraria a la vacuna contra el VPH (Antivacuna).  
        1: Postura a favor de la vacuna contra el VPH (Provacuna).  
        2: Expresa dudas relacionadas con la vacuna contra el VPH.  
        3: Comentarios que no se relacionan con la vacuna contra el VPH.  
    """
    )

    try:
    # Botón para ocultar/mostrar la API de OpenAI
        api_key = st.text_input("API Key de OpenAI", type="password")
    # Mostrar advertencia si no se ha ingresado la API Key
        if not api_key:
            st.warning("Ingrese su API Key de OpenAI.")
            return
    except openai.APIError as e:
    # Manejar error de la API aquí, por ejemplo, reintentar o registrar
            st.error(f"La API de OpenAI devolvió un Error de API: {e}")
            pass

    except openai.APIConnectionError as e:
    # Manejar error de conexión aquí
        st.error(f"No se pudo conectar a la API de OpenAI: {e}")
        pass

    except openai.RateLimitError as e:
    # Manejar error de límite de velocidad (recomendamos usar un retraso exponencial)
        st.error(f"La solicitud a la API de OpenAI excedió el límite de velocidad: {e}")
        pass
    except Exception as e:
    # Manejar cualquier excepción que ocurra aquí
        st.error(f"Se produjo un error al procesar la API Key: {e}")
                      
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
            
            column_name = st.text_input("Nombre de la columna:")
            if not column_name:
                st.warning("Ingrese el nombre de la columna que contiene los comentarios.")
                return
            if column_name not in data.columns:
                st.write("Error: La columna especificada no existe en los datos. Por favor ingrese el nombre de la columna  que contiene los comentarios")
                return None
            
           
                
            # Clasificar los comentarios si se ha proporcionado la API Key
            if api_key:
                #openaiapi_key="'"+ str(api_key) + "'"
                X_nuevos = obtener_incrustaciones(data, column_name, api_key)
                #st.write(X_nuevos)
                # Añadir ceros adicionales para igualar el número de características esperado por el modelo
                #X_nuevos_con_padding = np.pad(X_nuevos, ((0, 0), (0, 22)), mode='constant')
                #st.write(X_nuevos_con_padding)
                modelo_cargado = pickle.loads(modelo())
                # Hacer predicciones con el modelo cargado utilizando los datos con padding
                #warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                predicciones_nuevas = modelo_cargado.predict(X_nuevos)
                #st.write(predicciones_nuevas)
                st.write("Datos clasificados:")
                # Agregar una nueva columna "Clasificación_gpt_4" con los valores de las predicciones
                data['Clasificación_gpt_4'] = predicciones_nuevas
                st.write(data)

        except openai.AuthenticationError as e:
            st.error("Error de autenticación: La clave de la API no es válida o ha expirado.")
            st.write("Por favor, asegúrate de que la clave de la API sea correcta y esté activa.")
            st.write("También verifica que estés siguiendo el formato correcto al proporcionar la clave de la API.")
        
        except Exception as e:
           st.error(f"Error al cargar el archivo: {e}")
            



    # Mostrar comentarios antivacunas al hacer clic en un botón
    if st.button("Mostrar comentarios antivacunas"):
        comentarios_antivacunas = data[data['Clasificación_gpt_4'] == 0][column_name].tolist()
        st.subheader("Comentarios antivacunas encontrados:")
        if comentarios_antivacunas:
            comentario2=[]
            for comentario in comentarios_antivacunas:
                #st.write(comentario)
                comentario2.append({'Comentarios antivacunas': comentario})
                
            df_comentario2 = pd.DataFrame(comentario2)
            # Mostrar el DataFrame con el comentario
            st.dataframe(df_comentario2)
            # Agregar el botón de descarga
            st.download_button(
            label="Descargar comentarios antivacunas CSV",
            data=download_csv(df_comentario2),
            file_name="comentarios_antivacunas.csv",
            mime="text/csv"
            )
        else:
            st.write("No se encontraron comentarios antivacunas.")

    # Mostrar comentarios antivacunas al hacer clic en un botón
    if st.button("Mostrar comentarios dudas"):
        comentarios_duda = data[data['Clasificación_gpt_4'] == 2][column_name].tolist()
        st.subheader("Comentarios de dudas:")
        if comentarios_duda:
            comentario1=[]
            for comentario in comentarios_duda:
                #st.write(comentario)
                # Crear un DataFrame con el comentario
                comentario1.append({'Comentarios dudas': comentario})
                
            df_comentario = pd.DataFrame(comentario1)
            # Mostrar el DataFrame con el comentario
            st.dataframe(df_comentario)
            # Agregar el botón de descarga
            st.download_button(
            label="Descargar comentarios dudas CSV",
            data=download_csv(df_comentario),
            file_name="comentarios_dudas.csv",
            mime="text/csv"
            )
        else:
            st.write("No se encontraron comentarios dudas.")

    

if __name__ == "__main__":
    run()
