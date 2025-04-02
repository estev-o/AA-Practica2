using FileIO
using Images

# Cargar una imagen
imagen = load("lena.jpg");
#Extraemos los canales
imagenRGB = RGB.(red.(imagen), green.(imagen), blue.(imagen))
# Convertir la imagen a escala de grises
imagenGris = Gray.(imagenRGB)
# Mostrar la imagen en escala de grises
display(imagenGris)