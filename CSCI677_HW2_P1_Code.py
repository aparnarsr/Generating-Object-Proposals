#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
import xml.etree.ElementTree as et


# In[ ]:


# img1=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/000480.jpg")
img1=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/001324.jpg")
# img1=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/002129.jpg")
# img1=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/002268.jpg")
# img1=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/003129.jpg")
# img1=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/004085.jpg")
# img1=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/004708.jpg")
# img1=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/007346.jpg")

# xtree = et.parse(os.getcwd()+"/Downloads/Annotations/000480.xml")
xtree = et.parse(os.getcwd()+"/Downloads/Annotations/001324.xml")
# xtree = et.parse(os.getcwd()+"/Downloads/Annotations/002129.xml")
# xtree = et.parse(os.getcwd()+"/Downloads/Annotations/002268.xml")
# xtree = et.parse(os.getcwd()+"/Downloads/Annotations/003129.xml")
# xtree = et.parse(os.getcwd()+"/Downloads/Annotations/004085.xml")
# xtree = et.parse(os.getcwd()+"/Downloads/Annotations/004708.xml")
# xtree = et.parse(os.getcwd()+"/Downloads/Annotations/007346.xml")
xroot = xtree.getroot()
df=[]
for dims in xroot.findall('object/bndbox'):
        xmin=int(dims.find('xmin').text)
        xmax=int(dims.find('xmax').text)
        ymin=int(dims.find('ymin').text)
        ymax=int(dims.find('ymax').text)
        df.append([xmin,xmax,ymin,ymax])
len(df)


# In[ ]:


src = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.addImage(src)
graph= cv2.ximgproc.segmentation.createGraphSegmentation()
ss.addGraphSegmentation(graph)
color=cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
size=cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
texture=cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
shape=cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
allstrategies=cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(color,size,texture,shape)
ss.addStrategy(color)
# ss.addStrategy(allstrategies)


# In[ ]:


def findIou(bA, bB):
    #Coordinates of the rectangle in the intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    #Intersection area
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # Proposal and ground truth area
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# In[ ]:


rects=ss.process()
print("Total Number of proposals",len(rects))
num=100


# In[ ]:


#Max value to determine highest iou and iou>0.5
maxi=0.5

#Counts to determine number of proposals>0.5
count=0
count0=0
count1=0

#Counts to determine Recall
c0=0
c1=0

while True:
    Prop=img1.copy() #All Proposals
    imOut = img1.copy() #All ground truths

    for i, rect in enumerate(rects):
        if (i < num): # For number of proposals defined-100 
            x, y, w, h = rect #Dimensions of proposal 
            
            #showing all proposal boxes
            cv2.rectangle(Prop, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            bA=(x,y,x+w,y+h) #Coordinates of current proposal box
        
            boxB0=(df[0][0],df[0][2],df[0][1],df[0][3]) #Coordinates of ground truth 0
            boxB1=(df[1][0],df[1][2],df[1][1],df[1][3]) #Coordinates of ground truth 1
            
            iou0=findIou(bA, boxB0) #Calculatng iou of current proposal box with ground truth 0
            iou1=findIou(bA, boxB1) #Calculatng iou of current proposal box with ground truth 1

            for i in range(len(df)):
                
                #Adding all ground truths and proposals with iou>0.5
                bB=(df[i][0],df[i][2],df[i][1],df[i][3])
                iou=findIou(bA, bB)
                if (iou > maxi):
                    count=count+1
                    cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            if (iou0 > maxi):
                count0=count0+1
            if (iou1 > maxi):
                count1=count1+1
        else:
            break  
            
    for i in range(len(df)):
        cv2.rectangle(imOut,((df[i][0]),(df[i][2])),((df[i][1]),(df[i][3])),(0,0,255),1,cv2.LINE_AA) #Image with all ground truths  
    
    cv2.imshow("Output", imOut) #Show Image with all ground truths and corresponding proposals with iou>0.5
    cv2.imshow("Proposals", Prop) #Show Image with all Proposals
    
    print("count ",count) #Number of proposals with iou>0.5
    print("count0 ",count0) #Number of proposals with iou>0.5 for ground truth0
    print("count1 ",count1) #Number of proposals with iou>0.5 for ground truth1

    if count0>0:
        c0=1;
    if count1>0:
        c1=1;

    Recall=(c0+c1)/len(df)
    print("Recall",Recall)
    print("Total number of proposals",len(rects))
    
    k = cv2.waitKey(0) & 0xFF
    if k == 113:
        break
    
cv2.destroyAllWindows()

