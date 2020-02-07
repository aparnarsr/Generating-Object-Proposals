#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import xml.etree.ElementTree as et
import os


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


# im=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/000480.jpg")
# im=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/001324.jpg")
im=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/002129.jpg")
# im=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/002268.jpg")
# im=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/003129.jpg")
# im=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/004085.jpg")
# im=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/004708.jpg")
# im=cv2.imread(os.getcwd()+"/Downloads/JPEGImages/007346.jpg")

# xtree = et.parse(os.getcwd()+"/Downloads/Annotations/000480.xml")
# xtree = et.parse(os.getcwd()+"/Downloads/Annotations/001324.xml")
xtree = et.parse(os.getcwd()+"/Downloads/Annotations/002129.xml")
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


model = os.getcwd()+'/Downloads/model.yml.gz'
src = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
edges = edge_detection.detectEdges(np.float32(src)/255) 
orimap = edge_detection.computeOrientation(edges)# orientation map
edges = edge_detection.edgesNms(edges, orimap)#Edge suppression
edge_boxes = cv2.ximgproc.createEdgeBoxes()
edge_boxes.setMaxBoxes(100)
edge_boxes.setBeta(0.5)
edge_boxes.setAlpha(0.5)
boxes, scores = edge_boxes.getBoundingBoxes(edges, orimap)
print("Number of boxes",len(boxes))


# In[ ]:


#Counts to determine number of proposals>0.5
count=0
count0=0
count1=0

#Counts to determine Recall
c0=0
c1=0

Prop=im.copy() #All proposals
imOut = im.copy() #Proposals>0.5

for b in boxes:
    x,y,w,h = b
    boxA=(x,y,x+w,y+h) # Coordinates of current proposal
    
    #Adding all proposals to Prop output
    cv2.rectangle(Prop, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    #Getting coordinates of ground truths
    boxB0=(df[0][0],df[0][2],df[0][1],df[0][3])
    boxB1=(df[1][0],df[1][2],df[1][1],df[1][3])
    
    #Calculating IOU of current proposal and ground truths
    iou0=bb_intersection_over_union(boxA, boxB0)
    iou1=bb_intersection_over_union(boxA, boxB1)

    if (iou0 > 0.5):
            count0=count0+1
            cv2.rectangle(imOut0, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    if (iou1 > 0.5):
            count1=count1+1
            cv2.rectangle(imOut1, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            
    for i in range(len(df)):#Iterating through all ground truths
        boxB=(df[i][0],df[i][2],df[i][1],df[i][3])#Calculating each ground truth area
        iou=bb_intersection_over_union(boxA, boxB) #Calculating iou with current proposal box
        if (iou > 0.5):
            count=count+1
            cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
for i in range(len(df)):
    cv2.rectangle(imOut,((df[i][0]),(df[i][2])),((df[i][1]),(df[i][3])),(0,0,255),1,cv2.LINE_AA)  

cv2.imshow("Output", imOut)
print("count ",count)
print("count0 ",count0)
print("count1 ",count1)

if count0>0:
    c0=1;
if count1>0:
    c1=1;

Recall=(c0+c1)/len(df)
print("Recall",Recall)

cv2.imshow("Proposals",Prop)
cv2.waitKey(0)

cv2.destroyAllWindows()

