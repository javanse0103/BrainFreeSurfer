import streamlit as st 
import nibabel as nib
import numpy as np 
import os
#from streamlit_image_select import image_select
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import warnings

st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")

# @st.cache_data#(allow_output_mutation=True)
# def load_mri_data(freesurfer_dir, subject_id):
#     paths = {
#         'scripts': os.path.join(freesurfer_dir, 'subjects', subject_id, 'scripts'),
#         'mri': os.path.join(freesurfer_dir, 'subjects', subject_id, 'mri'),
#         'surf': os.path.join(freesurfer_dir, 'subjects', subject_id, 'surf'),
#         'label': os.path.join(freesurfer_dir, 'subjects', subject_id, 'label'),
#         'stats': os.path.join(freesurfer_dir, 'subjects', subject_id, 'stats')
#     }
    
#     files = {
#         'scripts_files': os.listdir(paths['scripts']),
#         'mri_files': os.listdir(paths['mri']),
#         'surf_files': os.listdir(paths['surf']),
#         'label_files': os.listdir(paths['label']),
#         'stats_files': os.listdir(paths['stats']),
#     }

#     data_files = {
#         'aseg': os.path.join(paths['mri'], 'aseg.mgz'),
#         'orig': os.path.join(paths['mri'], 'orig.mgz'),
#         'T1': os.path.join(paths['mri'], 'T1.mgz'),
#         'brainmask': os.path.join(paths['mri'], 'brainmask.mgz'),
#         'wm': os.path.join(paths['mri'], 'wm.mgz')
#     }

#     data = {
#         'aseg': nib.load(data_files['aseg']).get_fdata(),
#         'orig': nib.load(data_files['orig']).get_fdata(),
#         'T1': nib.load(data_files['T1']).get_fdata(),
#         'brainmask': nib.load(data_files['brainmask']).get_fdata(),
#         'wm': nib.load(data_files['wm']).get_fdata(),
#         "stats":os.path.join(paths["stats"],"aseg.stats")
#     }
           

#     return paths, files, data

#@st.cache_data
@st.cache
def asegg(data,colormap):
    if colormap is not None:
         
    # Leer el archivo LUT y crear un diccionario para almacenar los colores
    #freesurfer_lut_path = os.path.join(freesurfer_dir, colormap)
        label_colors = {}
        lines = colormap.split('\n')
        for line in lines:
                if not line.startswith('#') and line.strip() != '':
                    parts = line.split()
                    label = int(parts[0])
                    structure_name = parts[1]
                    r, g, b = int(parts[2]), int(parts[3]), int(parts[4])
                    # Asegúrate de almacenar un diccionario para cada etiqueta
                    label_colors[label] = {'structure_name': structure_name, 'color': (r , g , b )}

        mapa_colores = np.zeros((256, 3))  
        for intensidad, valor in label_colors.items():
            mapa_colores[intensidad] = np.array(valor['color']) / 255 

        aseg_data = np.array(data)
        imagenes_coloreadas = mapa_colores[aseg_data.astype(int)]
        return(imagenes_coloreadas)

if 'LUT' in st.session_state and st.session_state.text_content is not None:
    colormap_content = st.session_state.text_content
    # Asegúrate de que 'aseg' está en st.session_state.volumes antes de llamar a la función
    if 'aseg' in st.session_state.volumes and st.session_state.volumes['aseg'] is not None:
        aseg_data_coloreada = asegg(st.session_state.volumes, colormap_content)
        # Aquí puedes hacer algo con aseg_data_coloreada, como mostrarla en la interfaz
#else:
#    st.error("No se ha cargado el contenido del archivo LUT.")

# Utiliza la función para cargar los datos
freesurfer_dir = '/Applications/freesurfer/7.4.1'
subject_id = 'bert'
#paths, files, mri_data= load_mri_data(freesurfer_dir, subject_id)


# Función para cargar datos de MRI
def load_mri_volume(uploaded_file):
    # Guardar el archivo temporalmente
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Cargar el archivo .mgz y obtener los datos
    volume_data = nib.load(uploaded_file.name).get_fdata()
    
    # Eliminar el archivo temporal
    os.remove(uploaded_file.name)
    
    return volume_data

# Inicializar session_state para los volúmenes de MRI
if 'volumes' not in st.session_state:
    st.session_state.volumes = {
        'aseg': None,
        'orig': None,
        'brainmask': None
    }
if 'LUT' not in st.session_state:
    st.session_state.text_content = None

if 'stats' not in st.session_state:
    st.session_state.stats = None    

with st.sidebar:
     
    st.header("Instrucciones")
    st.write("""
        Añadir los siguientes archivos en este orden para el análisis:
        - `orig.mgz`: Imagen original
        - `aseg.mgz`: Segmentaciones subcorticales
        - `brainmask.mgz`: Máscara del cerebro
        - `wm.mgz`: Segmentación de la materia blanca
        - `AsegStatsLUT.txt`: Mapa de colores para `aseg.mgz`
        - `aseg.stats`: Estadísticas de las segmentaciones
    """)
    uploaded_files = st.file_uploader("", accept_multiple_files=True,
    key="file_uploader")


if uploaded_files:
    for uploaded_file in uploaded_files:
        # Cargar y procesar cada archivo .mgz
        if uploaded_file.name.endswith('.mgz'):
            volume_data = load_mri_volume(uploaded_file)
            
            if 'aseg' in uploaded_file.name:
                st.session_state.volumes['aseg'] = volume_data
            elif 'orig' in uploaded_file.name:
                st.session_state.volumes['orig'] = volume_data
            elif 'brainmask' in uploaded_file.name:
                st.session_state.volumes['brainmask'] = volume_data
            elif 'wm' in uploaded_file.name:
                st.session_state.volumes['wm'] = volume_data    

        elif uploaded_file.name.endswith('.txt'):
            st.session_state.text_content = uploaded_file.getvalue().decode("utf-8")

        else:
             st.session_state.stats = uploaded_file    

imagenes_coloreadas = asegg(st.session_state.volumes['aseg'],st.session_state.text_content)


#@st.cache_data
@st.cache
def dataframe(data):
     
    stats_data = pd.read_csv(data, comment='#', delim_whitespace=True,
                            names=['Index', 'SegId', 'NVoxels', 'Volume_mm3', 
                                    'StructName', 'normMean', 'normStdDev', 
                                    'normMin', 'normMax', 'normRange'])
    return stats_data


####################################################
############### interfaz streamlit #################
####################################################

st.markdown("<h1 style='text-align: center; font-weight: bold;'>Brain MRI</h1>", unsafe_allow_html=True)
st.markdown("---")


def mostrar_cuatro_imagenes_rotadas(volumen, n_corte):

    if st.session_state.volumes['orig'] is not None:
    
        if skullstrip:
                
                brainmask_axial = np.rot90(st.session_state.volumes['brainmask'][:, :, n_corte], k=-1)
                axes[0,0].imshow(brainmask_axial,cmap="gray",alpha=1)
                axes[0,0].axis('off')
                brainmask_sagital = np.rot90(st.session_state.volumes['brainmask'][:, n_corte, :], k=1)
                axes[0,1].imshow(brainmask_sagital,cmap="gray",alpha=1)
                axes[0,1].axis('off')
                axes[1,0].imshow(st.session_state.volumes['brainmask'][n_corte, :, :],cmap="gray",alpha=1)
                axes[1,0].axis('off')
                imagen_negra = np.rot90(np.zeros_like(st.session_state.volumes['brainmask'][:, :, n_corte]), k=-1)
                axes[1,1].imshow(imagen_negra, cmap="gray",alpha=1)
                axes[1,1].axis('off')
                
        else:
                # axial
                imagen_axial = np.rot90(st.session_state.volumes['orig'][:, :, n_corte], k=-1)
                axes[0,0].imshow(imagen_axial, cmap="gray",alpha=1)
                axes[0,0].axis('off')
                
                # sagital - Rotar 90 grados en sentido horario
                imagen_sagital = np.rot90(st.session_state.volumes['orig'][:, n_corte, :], k=1)
                axes[0,1].imshow(imagen_sagital, cmap="gray",alpha=1)
                axes[0,1].axis('off')

                #coronal 
                axes[1,0].imshow(st.session_state.volumes['orig'][n_corte, :, :], cmap="gray",alpha=1)
                axes[1,0].axis('off')
                
                # Cuarta imagen (en negro)
                imagen_negra = np.rot90(np.zeros_like(st.session_state.volumes['orig'][:, :, n_corte]), k=-1)
                axes[1,1].imshow(imagen_negra, cmap="gray",alpha=1)
                axes[1,1].axis('off')
                

        if aseg_segmentation:
                aseg_axial = np.rot90(imagenes_coloreadas[:, :, n_corte], k=-1)
                axes[0,0].imshow(aseg_axial,alpha=opac_aseg)
                axes[0,0].axis('off')
                aseg_sagital = np.rot90(imagenes_coloreadas[:, n_corte, :], k=1)
                axes[0,1].imshow( aseg_sagital ,alpha=opac_aseg)
                axes[0,1].axis('off')
                axes[1,0].imshow(imagenes_coloreadas[n_corte, :, :],alpha=opac_aseg)
                axes[1,0].axis('off')
                
        if wm_segmentation:
                wm_axial = np.rot90(st.session_state.volumes['wm'][:, :, n_corte], k=-1)
                axes[0,0].imshow(wm_axial,cmap="gray",alpha=opac_wm)
                axes[0,0].axis('off')
                wm_sagital = np.rot90(st.session_state.volumes['wm'][:, n_corte, :], k=1)
                axes[0,1].imshow(wm_sagital,cmap="gray",alpha=opac_wm)
                axes[0,1].axis('off')
                axes[1,0].imshow(st.session_state.volumes['wm'][n_corte, :, :],cmap="gray",alpha=opac_wm)
                axes[1,0].axis('off')

    return fig

###################################################################


#col1, col2 = st.columns([0.20,0.80])

#####################################################################

#with col1:

 #   st.markdown("---")
 #   st.markdown("Segmentations")
 #   skullstrip = st.checkbox('Skull stripping')
 #   wm_segmentation = st.checkbox("White matter segmentation")
 #   if wm_segmentation:
 #        opac_wm = st.slider("Opacity WM", 0.2, 0.7)
 #   aseg_segmentation = st.checkbox("subcortical structures segmentation")
 #   if aseg_segmentation:
 #        opac_aseg = st.slider("Opacity aseg", 0.2, 0.7)
 #   st.markdown("---")

 #   st.markdown("Slices")
 #   slice_axial = st.slider("Slices", 0, 255)

#with col2:
 #   col3, col4 = st.columns([0.8, 0.5])

  #  with col3:
   #     fig, axes = plt.subplots(2, 2, figsize=(10,10), constrained_layout=True)
    #    st.pyplot(mostrar_cuatro_imagenes_rotadas(st.session_state.volumes['orig'], slice_axial))        

    #with col4:
     #   if st.session_state.stats is not None:
      #      st.markdown("Volumes of subcortical structures")
            #st.dataframe(dataframe(mri_data["stats"])[['StructName', 'Volume_mm3']],width=400, height=665)
       #     st.dataframe(dataframe(st.session_state.stats)[['StructName', 'Volume_mm3']],width=400, height=665)
            #st.dataframe(df.style.apply(apply_color, axis=1))

# Columna principal 1
col1 = st.columns([0.20])

with col1:
    st.markdown("---")
    st.markdown("Segmentations")
    skullstrip = st.checkbox('Skull stripping')
    wm_segmentation = st.checkbox("White matter segmentation")
    if wm_segmentation:
        opac_wm = st.slider("Opacity WM", 0.2, 0.7)
    aseg_segmentation = st.checkbox("subcortical structures segmentation")
    if aseg_segmentation:
        opac_aseg = st.slider("Opacity aseg", 0.2, 0.7)
    st.markdown("---")
    st.markdown("Slices")
    slice_axial = st.slider("Slices", 0, 255)

# Columnas principales 2 y 3 (anidadas dentro de una fila)
row2 = st.container()

with row2:
    col2, col3 = st.columns([0.8, 0.5])

    with col2:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
        st.pyplot(mostrar_cuatro_imagenes_rotadas(st.session_state.volumes['orig'], slice_axial))

    with col3:
        if st.session_state.stats is not None:
            st.markdown("Volumes of subcortical structures")
            st.dataframe(dataframe(st.session_state.stats)[['StructName', 'Volume_mm3']], width=400, height=665)
