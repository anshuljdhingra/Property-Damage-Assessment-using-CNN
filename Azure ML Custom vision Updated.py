
# coding: utf-8

# In[34]:


import requests
import os
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt

def analyse_roof(image_data):
    #image_data is image in a byte array format.
    prediction_url="https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/35e146a5-6c86-42b7-90b6-7e35e9cbf954/image?iterationId=f8385b68-1e98-4e36-b2ed-c5e7dd9ac638"
    prediction_key="ca3ad3f6b67844a6b07ab9c119e1c89b"

    headers  = {'Prediction-Key': prediction_key, "Content-Type": "application/octet-stream"}

    response = requests.post(prediction_url, headers=headers, data=image_data)
    response.raise_for_status()
    analysis = response.json()
    
    # read raw byte array of image into opencv n-dimensional numpy array format for image manupulations  
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #calculate height and width of the image 
    img_height = np.size(img, 0)
    img_width = np.size(img, 1)
    
    ## assuming that 100 pixels in these images are equal to 0.8 metres
    pixels_per_metric = 100 / 0.8  # 125 pixels means 0.8 metres
    #print(pixels_per_metric)
    
    roof_flag=0  #0 for good roof  # 1 for Damaged roof # 2 means tarped roof 
    i=0
    total_damaged_area_in_pixels=0 #Damaged Area

    for x in analysis['predictions']:
        if (x['tagName'] == 'Damaged Roof') & (x['probability'] >= 0.055): #threshold value for probability.
            roof_flag=1  
            #print('next predicion')
            #print(analysis['predictions'][i])
            left=int(x['boundingBox']['left']*img_width)
            top=int(x['boundingBox']['top']*img_height)
            width=int(x['boundingBox']['width']*img_width)
            height=int(x['boundingBox']['height']*img_height)
            #find height and width in metres. using pixels_per_metric
            width_m=(x['boundingBox']['width']*img_width)/pixels_per_metric
            height_m=(x['boundingBox']['height']*img_height)/pixels_per_metric
            area=width_m*height_m
            total_damaged_area_in_pixels+=area
            #print(area)
            color=[0,0,255]
            cv2.rectangle(img, (left, top), (left+width, top+height), color, 2)
            
        elif (x['tagName'] == 'Tarped Roof') & (x['probability'] >= 0.055): #threshold value for probability.
            roof_flag=2  
            #print('Tarped roof')
            #print(analysis['predictions'][i])
            left=int(x['boundingBox']['left']*img_width)
            top=int(x['boundingBox']['top']*img_height)
            width=int(x['boundingBox']['width']*img_width)
            height=int(x['boundingBox']['height']*img_height)
            #find height and width in metres. using pixels_per_metric
            width_m=(x['boundingBox']['width']*img_width)/pixels_per_metric
            height_m=(x['boundingBox']['height']*img_height)/pixels_per_metric
            area=width_m*height_m
            total_damaged_area_in_pixels+=area
            #print(area)
            color=[0,0,255]
            cv2.rectangle(img, (left, top), (left+width, top+height), color, 2)

    if(roof_flag==1):
        print('Categorized as a damaged roof')
        print('Damaged Area in the roof: {0} metre sq. (Approx)'.format(int(total_damaged_area_in_pixels)))
        d_type='damaged roof'
    elif(roof_flag==2):
        print('Categorized as a Tarped roof')
        print('Tarped Area in the roof: {0} metre sq. (Approx)'.format(int(total_damaged_area_in_pixels)))
        d_type='tarped roof'
    else:
        print('Categorized as a Good roof')
        d_type='good roof'

    total_area=(img_width/pixels_per_metric)*(img_height/pixels_per_metric)
    #print('Total area of the roof is: {0} metre sq. '.format(total_area))

    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])  

    plt.imshow(rgb_img, cmap = 'gray', interpolation = 'bicubic')
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    cv2.imwrite('C:\Work\Project_1\Roof Result images\result.jpg',img)
    roof_analysis={}
    roof_analysis['Label']='Roof'
    roof_analysis['Category']=d_type
    roof_analysis['Damaged Area']=total_damaged_area_in_pixels
    roof_analysis['Result_path']='C:\Work\Project_1\Roof Result images\result.jpg'
    #instead of path for urgent solution, we can send output image as byte array in json data only, 
    
    return_output=json.dumps(roof_analysis, ensure_ascii=False)
    return return_output

#################################################################################################################################
#Main Program to call the function with image as byte array. (Binary Data)

filepath="C:\\Users\\isha.lamba\\OneDrive - NIIT Technologies\\AI POC Photos\\Roof test images\\Damaged roof 24.jpg"
image_data = open(filepath, "rb").read() #Read a file in binary format (array of Bytes)

roof_type=analyse_roof(image_data)
print(roof_type)

