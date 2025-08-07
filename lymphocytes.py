import os
import numpy as np
import cv2 as cv
import time 

def Seg_Cell_Nuclei(img_DAPI, params_seg):
    #img_DAPI: img en escala de grises
    #params_seg: diccionario de parámetros segmentación
    dFiltro = params_seg['dFiltro']
    sColor = params_seg['sColor']
    sSpace = params_seg['sSpace']
    se_close= params_seg['se_close']
    num_dilate = params_seg['num_dilate']
    se_dilation= params_seg['se_dilation']
    th_dt= params_seg['th_dt']
    se_erosion= params_seg['se_erosion']
    se_tophat= params_seg['se_tophat']
    se_cleaning= params_seg['se_cleaning']
    
    
    img_DAPI_copy=img_DAPI.copy()
    img_DAPI_copy = cv.cvtColor(img_DAPI_copy, cv.COLOR_GRAY2BGR)

    # Suavizado para reducir ruido
    img_DAPI = cv.bilateralFilter(img_DAPI, d=dFiltro, sigmaColor=sColor, sigmaSpace=sSpace)
 
    # Detección de bordes más robusta
    edges = cv.Canny(img_DAPI, 50, 150)

    # Top-hat
    se=cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_tophat, se_tophat))
    img_DAPI=cv.morphologyEx(img_DAPI, cv.MORPH_TOPHAT, se)
    
    # Umbralización adaptativa
    img_DAPI = cv.adaptiveThreshold(img_DAPI, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                 cv.THRESH_BINARY, blockSize=9, C=-1)

    # Operaciones morfológicas para mejorar la segmentación
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_close, se_close))
    img_DAPI = close_reconstruction(img_DAPI, se, 1)
    
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_dilation,se_dilation))
    img_DAPI=dilation(img_DAPI, se,num_dilate)

    # Relleno de agujeros
    img_DAPI =imfill(img_DAPI,5)

    # Transformada de distancia
    DAPI_distance=cv.distanceTransform(img_DAPI, cv.DIST_L2,3)
    dt_DAPI = cv.normalize(DAPI_distance,None,alpha=0,beta=255, 
                           norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)
    
    '''
    Esta parte se utilizó para encontrar visualmente el valor adecuado de threshold
    
    if show==True:
        cv.imshow('DAPI distance normalize', dt_DAPI)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite('DAPI distance normalize.png',dt_DAPI)

    val_thr = 0 
    def onchange_th(val): 
        global val_thr 
        val_thr= val 
        onchange_any() 
 
    def onchange_any(): 
        global dt_DAPI
        global thr_track_name
        global window_name 
        global val_thr
        realval_thr = val_thr
    
        retval, DAPI_dt_tmp=cv.threshold(dt_DAPI, realval_thr, 255, cv.THRESH_BINARY)
        if show==True:
            cv.imshow(window_name, DAPI_dt_tmp) 

        if show==True:
            thr_track_name="TH" 
            window_name='Threshold' 
            cv.namedWindow(window_name) 
            cv.createTrackbar(thr_track_name,window_name,0,255,onchange_th) 
            cv.waitKey(0)
            cv.destroyAllWindows() 
            #sacamos el valor del Threshold usando la barra
    '''
    # Umbralización
    retval, DAPI_threshold_dt = cv.threshold(dt_DAPI, th_dt, 255, cv.THRESH_BINARY)

    # Operaciones morfológicas posteriores
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_erosion,se_erosion))
    DAPI_threshold_dt=erosion(DAPI_threshold_dt, se, 1)

    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_close,se_close))
    DAPI_threshold_dt=close_reconstruction(DAPI_threshold_dt, se, 1)
    
    # Componentes conectadas
    num_labels, labels=cv.connectedComponents(DAPI_threshold_dt)
    nuclei_labelmap=np.uint8(labels)

    # Limpieza final de la imagen y los bordes
    nuclei_labelmap=Cleaning_Labelmap(nuclei_labelmap, se_cleaning)

    img_DAPI=cv.cvtColor(img_DAPI, cv.COLOR_GRAY2BGR)

    return nuclei_labelmap

def Det_CD8_Cells(img_CD8,nuclei_labelmap, params_det):
    #img_CD8: imagen en escala de grises
    #nuclei_label_map: resultado de seg_cell_nuclei
    #params_det: diccionario de parametros de detección
    
    se_tophat=params_det['se_tophat']
    c_th=params_det['c_th']
    se_dilate=params_det['se_dilate']
    num_dilate=params_det['num_dilate']
    se_erosion = params_det['se_erosion']
    num_erosion=params_det['num_erosion']
    se_open = params_det['se_open']
    num_open=params_det['num_open']

    # Laplaciano del gaussiano
    img_CD8=cv.GaussianBlur(img_CD8, (1,1),0)
    img_CD8=cv.Laplacian(img_CD8, cv.CV_32F, ksize=3)
    
    img_CD8 = cv.normalize(img_CD8, None, alpha=0,beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    img_CD8 = (255*img_CD8).astype(np.uint8)

    # Top-hat
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_tophat, se_tophat))
    img_CD8=cv.morphologyEx(img_CD8, cv.MORPH_TOPHAT, se)

    # Adaptative threshold
    CD8_th= cv.adaptiveThreshold(img_CD8, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                 cv.THRESH_BINARY, blockSize=7, C=c_th)
 
    # Operaciones morfológicas para mejorar la segmentación
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_dilate,se_dilate))
    CD8_th=dilation(CD8_th, se,num_dilate) 
    
    # Relleno de agujeros
    CD8_th =imfill(CD8_th,4)

    # Operaciones morfológicas posteriores
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_erosion,se_erosion))
    CD8_th=erosion(CD8_th, se, num_erosion)

    se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (se_open, se_open))
    CD8_th = open_reconstruction(CD8_th, se, num_open)

    # Transformar el nucleilabel map (obtenido en la función anterior) a binario
    # 0 el fondo y 1 los núcleos de células detectados
    retval,nuclei_labelmap=cv.threshold(nuclei_labelmap, 1, 255, cv.THRESH_BINARY)

    # Aplicar la detección de células CD8+ al nucleilabel map 
    nuclei_labelmap_CD8 = cv.bitwise_and(CD8_th,nuclei_labelmap)

    # Componentes conectadas
    n_CD8cells, _=cv.connectedComponents(nuclei_labelmap_CD8)
    # Para el número de células que presentan CD8+T-cells hay que quitar el label del fondo
    return n_CD8cells-1
#para evaluar la segmentación también se puso como output el CD8_th y así compararlo con el groundtruth de CD8,
#se ha quitado en el código a entregar para concordar con los output marcados

def Analyze_set(general_configuration):
    general_results=[]

    imgDAPI_path= general_configuration['imgDAPI_path']
    gtNuclei_path=general_configuration['gtNuclei_path']
    imgCD8_path=general_configuration['imgCD8_path']
    gtCD8_path=general_configuration['gtCD8_path']
    out_path=general_configuration['out_path']
    params_det=general_configuration['params_det']
    params_seg=general_configuration['params_seg']

    project_path = os.path.dirname(__file__) 
    os.chdir(project_path) 
    folder = os.listdir('lympho_database_v20241117/database_v20241117')
    folder.sort()
    DAPI_files = sorted(os.listdir(imgDAPI_path))
    CD8_files = sorted(os.listdir(imgCD8_path))
    gtNuclei_files=sorted(os.listdir(gtNuclei_path))
    gtCD8_files=sorted(os.listdir(gtCD8_path))
    out_files=sorted(os.listdir(out_path))

    # Parámetros para autoevaluarse
    max_dice =-1
    best_image_dice = None
    min_det_error=10

    for i in range(1,22):
        if i<10:
            image_ID = 'cutout_0'+str(i)
        else:
            image_ID='cutout_'+str(i)

        ruta_imagen_DAPI = os.path.join(imgDAPI_path,f"{image_ID}.tif")
        img_DAPI=cv.imread(ruta_imagen_DAPI,cv.IMREAD_ANYDEPTH)
        img_DAPI = cv.normalize(img_DAPI,None,0,255,cv.NORM_MINMAX).astype('uint8')
        ruta_imagen_gt_DAPI = os.path.join(gtNuclei_path,f"{image_ID}.tif")
        img_DAPI_gt=cv.imread(ruta_imagen_gt_DAPI,cv.IMREAD_ANYDEPTH)
        ruta_imagen_CD8 = os.path.join(imgCD8_path,f"{image_ID}.tif")
        img_CD8=cv.imread(ruta_imagen_CD8,cv.IMREAD_ANYDEPTH)
        ruta_imagen_gt_CD8 = os.path.join(gtCD8_path,f"{image_ID}.tif")
        img_CD8_gt=cv.imread(ruta_imagen_gt_CD8,cv.IMREAD_ANYDEPTH)

        t=time.process_time()
        # Segmentación
        nuclei_labelmap=Seg_Cell_Nuclei(img_DAPI, params_seg)
        n_CD8cells= Det_CD8_Cells(img_CD8,nuclei_labelmap,params_det)

        exec_time = time.process_time() - t 
        #print(f'time: {exec_time:.12f}')
        
        #dice=Evaluate_Segmentation(mask,img_CD8_gt)
        #print('dice mask: ')
        
        total_dice=Evaluate_Segmentation(nuclei_labelmap, img_DAPI_gt)
        #print(total_dice)

        det_error=Evaluate_Detection(n_CD8cells,img_CD8_gt)
        #print(det_error)

        results = {'image_ID':image_ID, 'total_dice':total_dice,'exec_time':exec_time,'det_error':det_error}  

        general_results.append(results)

        # Autoevaluación
        if total_dice>max_dice:
            max_dice = total_dice
            best_image_dice=image_ID
        if det_error<min_det_error:
            min_det_error=det_error
        
        # Crear el fuse labelmap y visualizarlo si True
        img_DAPI=cv.cvtColor(img_DAPI, cv.COLOR_GRAY2BGR)
        false_color_list=create_random_LUT(num_channels=3)
        fus_img, key_pressed = Fuse_Labelmap(img_DAPI, nuclei_labelmap, 
                                             false_color_list, 40, False, image_ID) 

    mean = 0
    y=0
    mean_time=0
    for r in general_results:
        dice=r['total_dice']
        error=r['det_error']
        times=r['exec_time']

        y+=error
        mean+=dice
        mean_time+=times
    

    mean = mean /21
    print('Mean dice: ', mean)

    y=y/53
    print('y: ',y)

    mean_time = mean_time /21
    print(f'Mean time: {mean_time:.10f}')

    # Autoevaluación
    print(f"El valor máximo de dice es: {max_dice:.4f}, obtenido por la imagen: {best_image_dice}")
    print("El valor mínimo de error de detección de CD8 es: ",min_det_error)
   
    return general_results

# ---------------------------------------------------------
#  Optional function
# ---------------------------------------------------------

def Cleaning_Labelmap(input_labelmap, params):
    # Border cleaning, dado que las células del borde no se puede saber si están enteras
    output_labelmap=imclearborder(input_labelmap)
    se=cv.getStructuringElement(cv.MORPH_ELLIPSE, (params,params))
    output_labelmap=open_reconstruction(output_labelmap, se, 1)

    return output_labelmap

# ---------------------------------------------------------
#  Segmentation functions
# ---------------------------------------------------------

# Hole filling
def imfill(image,se):
    aux_f = np.zeros(image.shape, dtype=np.uint8)
    aux_f[0,:] = aux_f[-1,:] = aux_f[:,0] = aux_f[:,-1] = 255
    image_inv=cv.bitwise_not(image)
    f=cv.bitwise_and(image_inv, aux_f)
    se_holes=cv.getStructuringElement(cv.MORPH_ELLIPSE, (se,se))
    result_inv=basic_reconstruction(f, image_inv, se_holes, 'dilate')
    result=cv.bitwise_not(result_inv)

    return result

# Basic reconstruction
def basic_reconstruction(marker, mask, se, method):
    prev_marker = marker
    new_marker = geo_step(prev_marker, mask,se, method)
    contador=0
    while not np.array_equal(prev_marker,new_marker):
        contador+=1
        prev_marker = new_marker
        new_marker = geo_step(prev_marker, mask, se, method)

    return new_marker

# Geo step
def geo_step (marker, mask, se, method):
    #if method is dilation or erosion perform the corresponding geodesic operation
    if method=='dilate':
        result = cv.bitwise_and(dilation (marker, se,1), mask)
        return result
    elif method=='erode':
        result = cv.bitwise_or(erosion(marker, se,1), mask)
        return result
    
# Erosionar n veces
def erosion(image,se,n):
    erosion=cv.erode(image,se,iterations=n)
    return erosion

# Dilatar n veces
def dilation(image,se,n):
    dilation=cv.dilate(image,se,iterations=n)
    return dilation

# Border clearing
def  imclearborder(image):
    aux_f = np.zeros(image.shape, dtype=np.uint8)
    aux_f[0,:] = aux_f[-1,:] = aux_f[:,0] = aux_f[:,-1] = 255
    f=cv.bitwise_and(image, aux_f)
    se_border=cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    #I-reconstruccion=a-b=a and (not b)
    b=basic_reconstruction(f, image, se_border, 'dilate')
    result=cv.bitwise_and(image, cv.bitwise_not(b))

    return result

# Open reconstruction
def open_reconstruction(image, se, n):
    #construct marker using erosion
    marker = erosion(image, se, n)
    result= basic_reconstruction(marker, image, se, 'dilate')

    return result

# Close reconstruction
def close_reconstruction(image, se, n):
    #construct marker using erosion
    marker = dilation(image, se, n)
    result= basic_reconstruction(marker, image, se, 'erode')

    return result


# ---------------------------------------------------------
#  Public functions
# ---------------------------------------------------------

def Evaluate_Segmentation(labelmap_test, labelmap_gt):
    # Function to evaluate a labelmap against a GT reference
    # The number of levels of the input labelmap is arbitrary 
    # There is no correspondence between labels of both input labelmaps
    # Background is always represented by 0 value
    # It returns a final Dice coefficient

    dice_value_direct = labelmap_DC(labelmap_gt, labelmap_test)
    dice_value_inverse = labelmap_DC(labelmap_test, labelmap_gt)
    out_dice = (dice_value_direct+dice_value_inverse)/2
    #print('   Final Dice value: %f ' % out_dice)
    return out_dice


def Evaluate_Detection(n_CD8cells, labelmap_gt_CD8):
    # Evaluate the number of detected CD8 cells in the gt labelmap
    num_labels_gt_cd8, _ = cv.connectedComponents(labelmap_gt_CD8)
    num_labels_gt_cd8 = num_labels_gt_cd8 - 1 # We remove the background label
    return np.abs(n_CD8cells - num_labels_gt_cd8)


def most_frequent_element(array,binary_mask):
    # Mask the array using the binary mask
    masked_vector = array[binary_mask==1]
    # Find the most frequent element within the masked region
    values, counts = np.unique(masked_vector, return_counts=True)
    return values[np.argmax(counts)] 

def labelmap_DC(labelmap_1, labelmap_2):
    # One-direction Dice coefficient
    # Label 0 must always mean background 

    labels_1 = np.unique(labelmap_1)
    num_labels_1 = len(labels_1)
    labels_2 = np.unique(labelmap_2)
    num_labels_2 = len(labels_2)

    if num_labels_1==1 and labels_1[0]==0: # Just background in labelmpap_1
        if num_labels_2==1 and labels_2[0]==0: # Both labelmaps are empty
            out_dice = 1
        else: # labelmpap_1 is empty and labelmap_2 is not
            out_dice = 0
    else: # labelmap_1 is not empty, it has some object instances
        if num_labels_2==1 and labels_2[0]==0: # labelmap_2 is empty
            out_dice = 0
        else: # Both labelmaps are not empty
            out_dice_vector = np.zeros(num_labels_1)
            for i in range(0,num_labels_1):
                if labels_1[i]!=0:
                    mask_i = np.uint8(0*labelmap_1)
                    mask_i[labelmap_1==labels_1[i]] = 1
                    label_most_freq = most_frequent_element(labelmap_2,mask_i)
                    mask_2 = 0*labelmap_2
                    mask_2[labelmap_2==label_most_freq] = 1

                    # f, axarr = plt.subplots(2,2)
                    # axarr[0,0].imshow(labelmap_1)
                    # axarr[0,1].imshow(mask_i)
                    # axarr[1,0].imshow(labelmap_2)
                    # axarr[1,1].imshow(mask_2)
                    # plt.show()

                    if label_most_freq==0:
                        out_dice_vector[i] = 0
                    else:
                        out_dice_vector[i] = binary_DC(mask_i, mask_2)
            if out_dice_vector[1:].size > 0:
                out_dice = out_dice_vector[1:].mean()
            else:
                out_dice = 0 
    return out_dice

def binary_DC(binary_img_1, binary_img_2):
    # Auxiliary function to calculate Dice coefficient from two binary images
    # The values of these MxN input images has to be 0 and 1, represented as uint8 type

    TP = np.count_nonzero(np.bitwise_and(binary_img_1, binary_img_2))
    TN = np.count_nonzero(np.bitwise_and(1-binary_img_1, 1-binary_img_2)) # Not used in this function. Included for completeness
    FP = np.count_nonzero(np.bitwise_and(binary_img_1, 1-binary_img_2))
    FN = np.count_nonzero(np.bitwise_and(1-binary_img_1, binary_img_2))
    dice_value = 2*TP/(2*TP+FP+FN)
    return dice_value

def create_random_LUT(num_channels=3):
    # Function to create a random false color list, used to represent a labelmap in a more appealing form
    # The output list will be used as input parameter in fusion_labelmap() function

    my_random_lut = np.zeros([256,1,num_channels],dtype=np.uint8)
    for i in range(num_channels):
        one_channel_lut = np.uint8(np.arange(256))
        np.random.shuffle(one_channel_lut[1:])
        my_random_lut[:,0,i] = one_channel_lut
    return my_random_lut

def Fuse_Labelmap(base_img, labelmap, false_color_list, alpha, showQ, window_title):
    # Function to fusion an input gray-level image, named as base image, 
    # with its corresponding segmentation labelmap, shown over this base image as a semitransparent color layer
    # Its input parameter are:
        # base_img: : MxN gray level input image that serves as background of the final fused result
        # labelmap: MxN image input labelmap to be overlaid with previous image
        # false_color_list: list of false colors used for coloring the input labelmap
        # alpha: value of alpha parameter controlling the transparency level of the input labelmap
        # showQ: boolean variable to control if a window is opened, showing the fused image, 
               # with a slider control to modify the transparency coefficient
        # window_title: the title for the opened window, when applicable
    # It returns the fusioned image and the code of the key pressed to close the window, 
    # if the input variable showQ was True

    base_img_norm_8U = cv.normalize(base_img,None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)
    base_img_rgb = base_img_norm_8U

    # Apply false colors to the given labels
    list_labels = np.unique(labelmap)
    [rows, cols] = labelmap.shape
    labelmap_false_color = np.zeros((rows,cols,3),dtype=np.uint8)
    for label_i in list_labels:
        labelmap_false_color[labelmap==label_i,:] = false_color_list[label_i]

    fus_img = cv.addWeighted(base_img_rgb,1-alpha,labelmap_false_color,alpha,0.0)
    if showQ:
        [rows, cols] = labelmap.shape
        width_window = 800
        cv.namedWindow(window_title, cv.WINDOW_NORMAL)
        cv.resizeWindow(window_title,width_window,round(width_window*rows/cols))
        alpha_slider_max = 100

        def onchange_any(val):
            val_alpha = cv.getTrackbarPos('Alpha blending coef', window_title)
            val_alpha = val_alpha/alpha_slider_max
            fus_img_aux = cv.addWeighted(base_img_rgb,1-val_alpha,labelmap_false_color,val_alpha,0.0)
            cv.imshow(window_title,fus_img_aux)

        cv.createTrackbar('Alpha blending coef',window_title,int(alpha*alpha_slider_max),
            alpha_slider_max,onchange_any)
        onchange_any(0)
        key_pressed = cv.waitKey()
        cv.destroyAllWindows()
    else:
        key_pressed = 0
    return fus_img, key_pressed
