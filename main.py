import lymphocytes as lymph
import cv2 as cv
import numpy as np
import copy as copy
import os

#Segmentación y detección CD8+ más óptima
print('• Resultados de la segmentación y detección CD8+ más óptima conseguida:')
general_configuration={'imgDAPI_path':"lympho_database_v20241117/database_v20241117/dapi"}
general_configuration['gtNuclei_path'] = os.path.join('lympho_database_v20241117', 'database_v20241117', 'dapi_mask')
general_configuration['imgCD8_path']   = os.path.join('lympho_database_v20241117', 'database_v20241117', 'cd8')
general_configuration['gtCD8_path']    = os.path.join('lympho_database_v20241117', 'database_v20241117', 'cd8_mask')
general_configuration['out_path']      = os.path.join('lympho_database_v20241117', 'database_v20241117', 'out')
general_configuration['params_seg']= {'dFiltro':7, 'sColor':60, 'sSpace':200,'se_tophat':34,'se_close':4,'num_dilate':1, 'se_dilation':3,'th_dt':15,'se_erosion':3,'se_cleaning':6}
general_configuration['params_det']={'se_tophat':4,'c_th':-2,'se_dilate':4, 'num_dilate':1,'se_erosion':9, 'num_erosion':1, 'se_open':3, 'num_open':3}    

general_results=lymph.Analyze_set(general_configuration)



#Pruebas de la sensibilidad del dice coefficient a cambios en params_seg
print('\n ---Segmentación (cambios en mean dice)---')
print('\n• Resultados segmentación si aumenta el tamaño del vencidario del filtro bilateral:')
general_configuration1=copy.deepcopy(general_configuration)
general_configuration1['params_seg']['dFiltro']=13

general_results1=lymph.Analyze_set(general_configuration1)

print('\n Se puede ver que disminuye la media dice de forma considerable.')

print('\n• Resultados segmentación si aumenta el tamaño del top-hat: ')
general_configuration1=copy.deepcopy(general_configuration)
general_configuration1['params_seg']['se_tophat']=90

general_results1=lymph.Analyze_set(general_configuration1)

print('\n Aunque nos alejemos mucho del valor propuesto como óptimo,')
print('no influye de forma significativa en la media dice.')

print('\n• Resultados segmentación si cambian los parámetros de las operaciones morfológicas (dilatación y erosión):')
print('\n Al variar el tamaño del elemento estructurante (dilatación):')
general_configuration1=copy.deepcopy(general_configuration)
general_configuration1['params_seg']['se_dilation']=4

general_results1=lymph.Analyze_set(general_configuration1)

print('\n Al variar el tamaño del elemento estructurante (erosión):')

general_configuration1=copy.deepcopy(general_configuration)
general_configuration1['params_seg']['se_erosion']=4

general_results1=lymph.Analyze_set(general_configuration1)

print('\n Al igual que ocurre en la dilatación, si aumentamos el tamaño del elemento estructurante en la erosión, disminuye la media dice.')
print('\n Al aumentar el número de repeticiones de estas operaciones morfológicas:')

general_configuration1=copy.deepcopy(general_configuration)
general_configuration1['params_seg']['num_dilate']=2

general_results1=lymph.Analyze_set(general_configuration1)

print('\n Tanto en la erosión como en la dilatación, la media dice se ve afectada bruscamente.')


#Pruebas de análisis de los params_det claves para mejorar la detección de CD8+
#(conseguir minimizar y)
print('\n ---Detección (cambios en y)---')
print('\n• Resultados detección de CD8+ si se disminuye el tamaño del top-hat:')
general_configuration5=copy.deepcopy(general_configuration)
general_configuration5['params_det']['se_tophat']=5

general_results5=lymph.Analyze_set(general_configuration5)

print('\nEl valor de y aumenta, pero no considerablemente, empeorando la detección')

print('\n• Resultados detección de CD8+ si la umbralización adaptativa tiene valor C positivo:')
general_configuration6=copy.deepcopy(general_configuration)
general_configuration6['params_det']['c_th']=2

general_results6=lymph.Analyze_set(general_configuration6)

print('\nEl parámetro y empeora de forma más considerable al hacer c positivo')
print('debido a que los pixeles necesitarán ser más oscuros para ser clasificados como fondo,')
print('incluyendo así menos detalles tenues.')

print('\n• Resultados detección de CD8+ si cambian las operaciones morfológicas (dilatación, erosión y apertura)')
print('\nSe observa que un ligero cambio en estos parámetros influye en menor medida.')
print('No obstante, un cambio excesivo de cualquiera de estos parámetros es determinante en la calidad de la detección.')

print('\nPor ejemplo, si se aumenta en 1 el elemento estructurante de la erosión: ')
general_configuration7=copy.deepcopy(general_configuration)
general_configuration7['params_det']['se_erosion']=10

general_results7=lymph.Analyze_set(general_configuration7)

print('\nEl detrimento de y no es tan significativo')

print('\n Sin embargo, un aumento excesivo de la repetición de esta erosión dará: ')
general_configuration8=copy.deepcopy(general_configuration)
general_configuration8['params_det']['num_erosion']=3

general_results8=lymph.Analyze_set(general_configuration8)

print('\nLa detección de células CD8+ empieza a estar bastante comprometida')
 
