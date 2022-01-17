import cv2
import numpy as np
import time
from scipy.spatial import distance_matrix


def mouseHandler(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print([x,y])

def heatmap(points, top):
    k = 21
    gauss = cv2.getGaussianKernel(k, np.sqrt(64))
    gauss = gauss * gauss.T
    gauss = gauss / gauss[k//2, k//2]
    spark = cv2.cvtColor(cv2.applyColorMap((gauss * 255).astype(np.uint8), cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB).astype(np.float32)/255

    heat = np.zeros(top.shape).astype(np.float32)

    for p in points:
        heat[p[1] - k // 2: 1 + p[1] + k // 2, p[0] - k // 2: 1 + p[0] + k // 2, :] += spark

    heat = heat / (np.max(heat, axis=(0, 1)) + 0.0001)
    gray = cv2.cvtColor(heat, cv2.COLOR_RGB2GRAY)
    mask = np.where(gray > 0.2, 1, 0).astype(np.float32)
    mask_3 = np.ones((top.shape[0], top.shape[1], 3)) * (1-mask)[:, :, None]
    mask_4 = heat * mask[:, :, None]
    new_top = (top * mask_3) + mask_4
    return new_top




def main():
    # Load Yolo
    net = cv2.dnn.readNet("yolov3.weights", "yolov3_big.cfg")
    net_mask = cv2.dnn.readNet("mask-yolov3_10000.weights", "mask-yolov3.cfg")
    classes = []
    classes_mask=[]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    with open("mask-obj.names", "r") as f1:
        classes_mask = [line1.strip() for line1 in f1.readlines()]
    layer_names = net.getLayerNames()
    layer_names_mask = net_mask.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    output_layers_mask=[layer_names_mask[i[0] - 1] for i in net_mask.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


    print("Press 1 for pre-recorded videos, 2 for live stream: ")
    option = int(input())

    if option == 1:
        # Record video
        windowName1= "Sample Feed from Camera 1"
        windowName2= "Sample Feed from Camera 2"
        windowName3= "Sample Feed from Camera 3"
        windowName4= "top view"
        
        cv2.namedWindow(windowName1)
        cv2.namedWindow(windowName2)
        cv2.namedWindow(windowName3)
        #cv2.namedWindow(windowName4)


     


        capture1 = cv2.VideoCapture("vid1.mp4")  # phone 1 camera
        capture2 = cv2.VideoCapture("vid2.mp4")   # laptop camera
        capture3 = cv2.VideoCapture("three.mp4") #phone 2 camera
        top_view=cv2.imread("top_view.jpeg")
        radius = 10
        color = (255, 0, 0)


     

        op_3=np.float32([[440, 147], [1830, 300], [1585, 820], [125, 670]])
        tp_3=np.float32([[470, 750], [150, 430], [355, 230], [510, 310]])




      

        op_1=np.asarray([[1395,252],[1104,267],[1544,435],[1891,333]],np.float32)
        tp_1=np.asarray([[388,113],[312,190],[469,363],[550,190]],np.float32)

       

        op_2=np.asarray([[304,251],[587,315],[1031,346],[723,539]],np.float32)
        tp_2=np.asarray([[551,191],[469,352],[470,512],[312,430]],np.float32)



        projective_matrix_3,_ = cv2.findHomography(op_3, tp_3)
        projective_matrix_1,_ = cv2.findHomography(op_1, tp_1)
        projective_matrix_2,_ = cv2.findHomography(op_2, tp_2)
        cv2.setMouseCallback(windowName1, mouseHandler)
        cv2.setMouseCallback(windowName2, mouseHandler)
        cv2.setMouseCallback(windowName3, mouseHandler)
        #cv2.setMouseCallback(windowName4, mouseHandler)


        

        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id_1 = 0
        frame_id_2 = 0
        frame_id_3 = 0
        # define size for recorded video frame for video 1
        width1 = int(capture1.get(3))
        height1 = int(capture1.get(4))
        size1 = (width1, height1)

        # define size for recorded video frame for video 2
        width2 = int(capture2.get(3))
        height2 = int(capture2.get(4))
        size2 = (width2, height2)

        # define size for recorded video frame for video 3
        width3 = int(capture3.get(3))
        height3 = int(capture3.get(4))
        size3 = (width3, height3)


        
        # frame of size is being created and stored in .avi file
        optputFile1 = cv2.VideoWriter(
            'cam1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
        optputFile2 = cv2.VideoWriter(
            'cam2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size2)
        optputFile3 = cv2.VideoWriter(
            'cam3.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size3)
        


        # check if feed exists or not for camera 1
        if capture1.isOpened():
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()





            
        else:
            ret1 = False
            ret2 = False
            ret3 = False
            
        frame_points = []
        offline_data = [[], ]
        offline_heatmap_points = [[], ]
        sop_data = [[], ]
        sop_heatmap_points=[[], ]
        while ret1 and ret2 and ret3:

            
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
            points=[]

            frame3_output = cv2.warpPerspective(frame3, projective_matrix_3, (864,932))

            
            frame1_output = cv2.warpPerspective(frame1, projective_matrix_1, (864,932))
            #cv2.imshow(windowName1, frame1_output)

            frame2_output = cv2.warpPerspective(frame2, projective_matrix_2, (864,932))


            
            mean=(np.mean([frame1_output,frame3_output],axis=0))/255.0
            
            
            summation=mean
            
            summation[np.logical_and(np.greater(frame1_output, 0),np.equal(frame2_output, 0),np.equal(frame3_output, 0))]=frame1_output[np.logical_and(np.greater(frame1_output, 0),np.equal(frame2_output, 0),np.equal(frame3_output, 0))]/255.0
            summation[np.logical_and(np.greater(frame2_output, 0),np.equal(frame1_output, 0),np.equal(frame3_output, 0))]=frame2_output[np.logical_and(np.greater(frame2_output, 0),np.equal(frame1_output, 0),np.equal(frame3_output, 0))]/255.0


            

            
            frame_id_1 += 1
            # Detecting objects
            blob_1 = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_1)
            net_mask.setInput(blob_1)
            outs = net.forward(output_layers)
            outs_mask=net_mask.forward(output_layers_mask)
            # Showing informations on the screen for person
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width1)
                        center_y = int(detection[1] * height1)
                        w = int(detection[2] * width1)
                        h = int(detection[3] * height1)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
            class_ids_mask = []
            confidences_mask = []
            boxes_mask = []
            for out_mask in outs_mask:
                for detection_mask in out_mask:
                    scores_mask = detection_mask[5:]
                    class_id_mask = np.argmax(scores_mask)
                    confidence_mask = scores_mask[class_id_mask]
                    if confidence_mask > 0.1:
                        # Object detected
                        center_x = int(detection_mask[0] * width1)
                        center_y = int(detection_mask[1] * height1)
                        w = int(detection_mask[2] * width1)
                        h = int(detection_mask[3] * height1)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes_mask.append([x, y, w, h])
                        confidences_mask.append(float(confidence_mask))
                        class_ids_mask.append(class_id_mask)

            indexes_mask = cv2.dnn.NMSBoxes(boxes_mask, confidences_mask, 0.4, 0.3)


            for i in range(len(boxes_mask)):
                    if i in indexes_mask:
                        x, y, w, h = boxes_mask[i]
                        label = str(classes_mask[class_ids_mask[i]])
                        
                        x, y, w, h = boxes_mask[i]
                        label = str(classes_mask[class_ids_mask[i]])
                        confidence_mask = confidences_mask[i]
                        color = colors[class_ids[i]]
                        cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
                        cv2.rectangle(frame1, (x, y), (x + w, y + 30), color, -1)
                        cv2.putText(frame1, label + " " + str(round(confidence_mask, 2)), (x, y + 30), font, 3, (255,255,255), 3)
                        mid_1=int((2*x+w)/2)
                        mid_2=int((2*y+2*h)/2)
                        pt_1=np.matmul(projective_matrix_1,[mid_1,mid_2,1])
                        pt_1=pt_1/pt_1[2]
                        cv2.circle(summation,(int(pt_1[0]),int(pt_1[1])), 10, color,-1)
                        

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    if (class_ids[i]==0):
                        confidence = confidences[i]
                        
                        mid_1=int((2*x+w)/2)
                        mid_2=int((2*y+2*h)/2)
                        cv2.circle(frame1,(mid_1,mid_2), 10, (0,0,255),-1)
                        pt_1=np.matmul(projective_matrix_1,[mid_1,mid_2,1])
                        
                        #print(pt_1/pt_1[2])
                        pt_1=pt_1/pt_1[2]
                        points.append((int(pt_1[0]),int(pt_1[1])))
                        cv2.circle(summation,(int(pt_1[0]),int(pt_1[1])), 10, (0,23,255),-1)
                        cv2.circle(frame1,(mid_1,mid_2), 10, (0,0,255),-1)

            dists=distance_matrix(points, points)
            
            
            #dists=np.asarray(dists)
            green=np.where(dists>80)
            #print(dists[green])
            g_1=green[0]
            g_2=green[1]
            for i in range(len(g_1)):
            
                cv2.circle(summation,points[g_1[i]], 10, (0,255,0),-1)
                cv2.circle(summation,points[g_2[i]], 10, (0,255,0),-1)

            red=np.where(dists<80)
            print(dists[red])
            r_1=red[0]
            r_2=red[1]
            red_points=[]
            for j in range(len(r_1)):
            
                cv2.circle(summation, points[r_1[i]], 10, (0,0,255),-1)
                cv2.circle(summation,points[r_2[i]], 10, (0,0,255),-1)
                red_points.append(points[r_1[i]])
                red_points.append(points[r_2[i]])
            
            sop_data[0] = red_points.copy()
            sop_heatmap_points[0].append(red_points)
            xx = heatmap(sum(sop_heatmap_points[0][-10:], []), top_view.copy()/255)   
            cv2.imshow('sopHeatmap', cv2.cvtColor((xx).astype(np.float32), cv2.COLOR_RGB2BGR))

            
            #print(green)




            elapsed_time = time.time() - starting_time
            fps = (frame_id_1 / elapsed_time)
            cv2.putText(frame1, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)

           
                    


            
            
            frame_id_2 += 1
            # Detecting objects
            blob_2 = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_2)
            net_mask.setInput(blob_2)
            outs = net.forward(output_layers)
            outs_mask=net_mask.forward(output_layers_mask)
            
            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width2)
                        center_y = int(detection[1] * height2)
                        w = int(detection[2] * width2)
                        h = int(detection[3] * height2)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
             # Showing informations on the screen for mask
            class_ids_mask = []
            confidences_mask = []
            boxes_mask = []
            for out_mask in outs_mask:
                for detection_mask in out_mask:
                    scores_mask = detection_mask[5:]
                    class_id_mask = np.argmax(scores_mask)
                    confidence_mask = scores_mask[class_id_mask]
                    if confidence_mask > 0.1:
                        # Object detected
                        center_x = int(detection_mask[0] * width2)
                        center_y = int(detection_mask[1] * height2)
                        w = int(detection_mask[2] * width2)
                        h = int(detection_mask[3] * height2)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes_mask.append([x, y, w, h])
                        confidences_mask.append(float(confidence_mask))
                        class_ids_mask.append(class_id_mask)

            indexes_mask = cv2.dnn.NMSBoxes(boxes_mask, confidences_mask, 0.4, 0.3)

            for i in range(len(boxes_mask)):
                    if i in indexes_mask:
                        x, y, w, h = boxes_mask[i]
                        label = str(classes_mask[class_ids_mask[i]])
                        
                        x, y, w, h = boxes_mask[i]
                        label = str(classes_mask[class_ids_mask[i]])
                        confidence_mask = confidences_mask[i]
                        color = colors[class_ids_mask[i]]
                        cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                        cv2.rectangle(frame2, (x, y), (x + w, y + 30), color, -1)
                        cv2.putText(frame2, label + " " + str(round(confidence_mask, 2)), (x, y + 30), font, 3, (255,255,255), 3)
                        
                        




            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    if (class_ids[i]==0):
                        confidence = confidences[i]
             
                        mid_1=int((2*x+w)/2)
                        mid_2=int((2*y+2*h)/2)
                        cv2.circle(frame2,(mid_1,mid_2), 10, (0,0,255),-1)
                        pt_2=np.matmul(projective_matrix_2,[mid_1,mid_2,1])
                        
                        #print(pt_1/pt_1[2])
                        pt_2=pt_2/pt_2[2]
                        #cv2.circle(summation,(int(pt_2[0]),int(pt_2[1])), 10, (0,23,255),-1)
                        points.append((int(pt_2[0]),int(pt_2[1])))
                        cv2.circle(frame2,(mid_1,mid_2), 10, (0,0,255),-1)



            elapsed_time = time.time() - starting_time
            fps = (frame_id_2 / elapsed_time)
            cv2.putText(frame2, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
                    

            
            frame_id_3 += 1
            # Detecting objects
            blob_3 = cv2.dnn.blobFromImage(frame3, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_3)
            net_mask.setInput(blob_3)
            outs_mask=net_mask.forward(output_layers_mask)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width3)
                        center_y = int(detection[1] * height3)
                        w = int(detection[2] * width3)
                        h = int(detection[3] * height3)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
               # Showing informations on the screen for mask
            class_ids_mask = []
            confidences_mask = []
            boxes_mask = []
            for out_mask in outs_mask:
                for detection_mask in out_mask:
                    scores_mask = detection_mask[5:]
                    class_id_mask = np.argmax(scores_mask)
                    confidence_mask = scores_mask[class_id_mask]
                    if confidence_mask > 0.1:
                        # Object detected
                        center_x = int(detection_mask[0] * width3)
                        center_y = int(detection_mask[1] * height3)
                        w = int(detection_mask[2] * width3)
                        h = int(detection_mask[3] * height3)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes_mask.append([x, y, w, h])
                        confidences_mask.append(float(confidence_mask))
                        class_ids_mask.append(class_id_mask)

            indexes_mask = cv2.dnn.NMSBoxes(boxes_mask, confidences_mask, 0.4, 0.3)

            for i in range(len(boxes_mask)):
                if i in indexes_mask:
                    x, y, w, h = boxes_mask[i]
                    label = str(classes_mask[class_ids_mask[i]])
                    
                    x, y, w, h = boxes_mask[i]
                    label = str(classes_mask[class_ids_mask[i]])
                    confidence_mask = confidences_mask[i]
                    color = colors[class_ids_mask[i]]
                    cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame3, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame3, label + " " + str(round(confidence_mask, 2)), (x, y + 30), font, 3, (255,255,255), 3)
                    
                    

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    if (class_ids[i]==0):
                        confidence = confidences[i]
           
                        #print(center_x,center_y)
                        mid_1=int((2*x+w)/2)
                        mid_2=int((2*y+2*h)/2)
                        cv2.circle(frame3,(mid_1,mid_2), 10, (0,0,255),-1)
                        pt_3=np.matmul(projective_matrix_3,[mid_1,mid_2,1])
                        
                        #print(pt_1/pt_1[2])
                        pt_3=pt_3/pt_3[2]
                        cv2.circle(summation,(int(pt_3[0]),int(pt_3[1])), 10, (0,23,255),-1)
                        cv2.circle(frame3,(mid_1,mid_2), 10, (0,0,255),-1)
                        points.append((int(pt_3[0]),int(pt_3[1])))





            elapsed_time = time.time() - starting_time
            fps = (frame_id_3 / elapsed_time)
            cv2.putText(frame3, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)

               
           
           
 



            #print((pt_1[0],pt_1[1]))
            #cv2.circle(summation,(int(pt_1[0]),int(pt_1[1])), 10, (0,23,255),-1)
            cv2.imshow('Final Result ', summation)

            cv2.imshow(windowName1, frame1)
            cv2.imshow(windowName2, frame2)
            cv2.imshow(windowName3, frame3)
            offline_data[0] = points.copy()
            offline_heatmap_points[0].append(points)
            final = heatmap(sum(offline_heatmap_points[0][-10:], []), top_view.copy()/255)
            frame_points.append(points)
            some = sum(frame_points, [])
            fin = heatmap(some, top_view/255)
            

          
            cv2.imshow('AnimatedHeatmap', cv2.cvtColor((final).astype(np.float32), cv2.COLOR_RGB2BGR))
            cv2.imshow('StaticHeatmap', cv2.cvtColor((fin).astype(np.float32), cv2.COLOR_RGB2BGR))


           

            # saves the frame from camera 1
            optputFile1.write(frame1)
            optputFile2.write(frame2)
            optputFile3.write(frame3)

            # escape key (27) to exit
            if cv2.waitKey(1) == 27:
                break
        capture1.release()
        cv2.destroyAllWindows()

        
    elif option == 2:
    
        # Record video
        windowName1= "Sample Feed from Camera 1"
        windowName2= "Sample Feed from Camera 2"
        windowName3= "Sample Feed from Camera 3"
        windowName4= "top view"
        
        cv2.namedWindow(windowName1)
        cv2.namedWindow(windowName2)
        cv2.namedWindow(windowName3)
        #cv2.namedWindow(windowName4)


     


        capture1 = cv2.VideoCapture('http://10.104.2.33:8080/video')  # phone 1 camera
        capture2 = cv2.VideoCapture(0)   # laptop camera
        capture3 = cv2.VideoCapture('http://10.104.2.54:8080/video') #phone 2 camera
        top_view=cv2.imread("top_view.jpeg")
        radius = 10
        color = (255, 0, 0)


        op_3=np.float32([[406, 780], [1063, 786], [1026, 86], [594, 688]])
        tp_3=np.float32([[629,351],[592,310],[556,353],[592,390]])




        op_1=np.asarray([[536,567],[753,460],[290,463],[518,363]],np.float32)
        tp_1=np.asarray([[473,589],[551,510],[393,511],[468,363]],np.float32)

       

        op_2=np.asarray([[651,783],[1131,545],[649,421],[371,481]],np.float32)
        tp_2=np.asarray([[309,427],[429 ,543],[541,429],[471,361]],np.float32)



        projective_matrix_3,_ = cv2.findHomography(op_3, tp_3)
        projective_matrix_1,_ = cv2.findHomography(op_1, tp_1)
        projective_matrix_2,_ = cv2.findHomography(op_2, tp_2)
        cv2.setMouseCallback(windowName1, mouseHandler)
        cv2.setMouseCallback(windowName2, mouseHandler)
        cv2.setMouseCallback(windowName3, mouseHandler)
        #cv2.setMouseCallback(windowName4, mouseHandler)


        

        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time()
        frame_id_1 = 0
        frame_id_2 = 0
        frame_id_3 = 0
        # define size for recorded video frame for video 1
        width1 = int(capture1.get(3))
        height1 = int(capture1.get(4))
        size1 = (width1, height1)

        # define size for recorded video frame for video 2
        width2 = int(capture2.get(3))
        height2 = int(capture2.get(4))
        size2 = (width2, height2)

        # define size for recorded video frame for video 3
        width3 = int(capture3.get(3))
        height3 = int(capture3.get(4))
        size3 = (width3, height3)


        
        # frame of size is being created and stored in .avi file
        optputFile1 = cv2.VideoWriter(
            'cam1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
        optputFile2 = cv2.VideoWriter(
            'cam2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size2)
        optputFile3 = cv2.VideoWriter(
            'cam3.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size3)
        


        # check if feed exists or not for camera 1
        if capture1.isOpened():
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()





            
        else:
            ret1 = False
            ret2 = False
            ret3 = False
            
        frame_points = []
        offline_data = [[], ]
        offline_heatmap_points = [[], ]
        sop_data = [[], ]
        sop_heatmap_points=[[], ]
        while ret1 and ret2 and ret3:

            
            ret1, frame1 = capture1.read()
            ret2, frame2 = capture2.read()
            ret3, frame3 = capture3.read()
            points=[]

            frame3_output = cv2.warpPerspective(frame3, projective_matrix_3, (864,932))

            
            frame1_output = cv2.warpPerspective(frame1, projective_matrix_1, (864,932))
            #cv2.imshow(windowName1, frame1_output)

            frame2_output = cv2.warpPerspective(frame2, projective_matrix_2, (864,932))


            
            mean=(np.mean([frame1_output,frame3_output],axis=0))/255.0
            
            
            summation=mean
            
            summation[np.logical_and(np.greater(frame1_output, 0),np.equal(frame2_output, 0),np.equal(frame3_output, 0))]=frame1_output[np.logical_and(np.greater(frame1_output, 0),np.equal(frame2_output, 0),np.equal(frame3_output, 0))]/255.0
            summation[np.logical_and(np.greater(frame2_output, 0),np.equal(frame1_output, 0),np.equal(frame3_output, 0))]=frame2_output[np.logical_and(np.greater(frame2_output, 0),np.equal(frame1_output, 0),np.equal(frame3_output, 0))]/255.0


            

            
            frame_id_1 += 1
            # Detecting objects
            blob_1 = cv2.dnn.blobFromImage(frame1, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_1)
            net_mask.setInput(blob_1)
            outs = net.forward(output_layers)
            outs_mask=net_mask.forward(output_layers_mask)
            # Showing informations on the screen for person
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width1)
                        center_y = int(detection[1] * height1)
                        w = int(detection[2] * width1)
                        h = int(detection[3] * height1)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
            class_ids_mask = []
            confidences_mask = []
            boxes_mask = []
            for out_mask in outs_mask:
                for detection_mask in out_mask:
                    scores_mask = detection_mask[5:]
                    class_id_mask = np.argmax(scores_mask)
                    confidence_mask = scores_mask[class_id_mask]
                    if confidence_mask > 0.1:
                        # Object detected
                        center_x = int(detection_mask[0] * width1)
                        center_y = int(detection_mask[1] * height1)
                        w = int(detection_mask[2] * width1)
                        h = int(detection_mask[3] * height1)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes_mask.append([x, y, w, h])
                        confidences_mask.append(float(confidence_mask))
                        class_ids_mask.append(class_id_mask)

            indexes_mask = cv2.dnn.NMSBoxes(boxes_mask, confidences_mask, 0.4, 0.3)


            for i in range(len(boxes_mask)):
                    if i in indexes_mask:
                        x, y, w, h = boxes_mask[i]
                        label = str(classes_mask[class_ids_mask[i]])
                        
                        x, y, w, h = boxes_mask[i]
                        label = str(classes_mask[class_ids_mask[i]])
                        confidence_mask = confidences_mask[i]
                        color = colors[class_ids[i]]
                        cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
                        cv2.rectangle(frame1, (x, y), (x + w, y + 30), color, -1)
                        cv2.putText(frame1, label + " " + str(round(confidence_mask, 2)), (x, y + 30), font, 3, (255,255,255), 3)
                        mid_1=int((2*x+w)/2)
                        mid_2=int((2*y+2*h)/2)
                        pt_1=np.matmul(projective_matrix_1,[mid_1,mid_2,1])
                        pt_1=pt_1/pt_1[2]
                        #cv2.circle(summation,(int(pt_1[0]),int(pt_1[1])), 10, color,-1)
                        

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    if (class_ids[i]==0):
                        confidence = confidences[i]
                        
                        mid_1=int((2*x+w)/2)
                        mid_2=int((2*y+2*h)/2)
                        cv2.circle(frame1,(mid_1,mid_2), 10, (0,0,255),-1)
                        pt_1=np.matmul(projective_matrix_1,[mid_1,mid_2,1])
                        
                        #print(pt_1/pt_1[2])
                        pt_1=pt_1/pt_1[2]
                        points.append((int(pt_1[0]),int(pt_1[1])))
                        #cv2.circle(summation,(int(pt_1[0]),int(pt_1[1])), 10, (0,23,255),-1)
                        cv2.circle(frame1,(mid_1,mid_2), 10, (0,0,255),-1)

            dists=distance_matrix(points, points)
            
            
            #dists=np.asarray(dists)
            green=np.where(dists>80)
            #print(dists[green])
            g_1=green[0]
            g_2=green[1]
            for i in range(len(g_1)):
            
                cv2.circle(summation,points[g_1[i]], 10, (0,255,0),-1)
                cv2.circle(summation,points[g_2[i]], 10, (0,255,0),-1)

            red=np.where(dists<80)
            print(dists[red])
            r_1=red[0]
            r_2=red[1]
            red_points=[]
            for j in range(len(r_1)):
            
                cv2.circle(summation, points[r_1[j]], 10, (0,0,255),-1)
                cv2.circle(summation,points[r_2[j]], 10, (0,0,255),-1)
                red_points.append(points[r_1[j]])
                red_points.append(points[r_2[j]])
            
            sop_data[0] = red_points.copy()
            sop_heatmap_points[0].append(red_points)
            xx = heatmap(sum(sop_heatmap_points[0][-2:], []), top_view.copy()/255)   
            cv2.imshow('sopHeatmap', cv2.cvtColor((xx).astype(np.float32), cv2.COLOR_RGB2BGR))

            
            #print(green)




            elapsed_time = time.time() - starting_time
            fps = (frame_id_1 / elapsed_time)
            cv2.putText(frame1, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)

           
                    


            
            
            frame_id_2 += 1
            # Detecting objects
            blob_2 = cv2.dnn.blobFromImage(frame2, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_2)
            net_mask.setInput(blob_2)
            outs = net.forward(output_layers)
            outs_mask=net_mask.forward(output_layers_mask)
            
            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width2)
                        center_y = int(detection[1] * height2)
                        w = int(detection[2] * width2)
                        h = int(detection[3] * height2)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
             # Showing informations on the screen for mask
            class_ids_mask = []
            confidences_mask = []
            boxes_mask = []
            for out_mask in outs_mask:
                for detection_mask in out_mask:
                    scores_mask = detection_mask[5:]
                    class_id_mask = np.argmax(scores_mask)
                    confidence_mask = scores_mask[class_id_mask]
                    if confidence_mask > 0.1:
                        # Object detected
                        center_x = int(detection_mask[0] * width2)
                        center_y = int(detection_mask[1] * height2)
                        w = int(detection_mask[2] * width2)
                        h = int(detection_mask[3] * height2)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes_mask.append([x, y, w, h])
                        confidences_mask.append(float(confidence_mask))
                        class_ids_mask.append(class_id_mask)

            indexes_mask = cv2.dnn.NMSBoxes(boxes_mask, confidences_mask, 0.4, 0.3)

            for i in range(len(boxes_mask)):
                    if i in indexes_mask:
                        x, y, w, h = boxes_mask[i]
                        label = str(classes_mask[class_ids_mask[i]])
                        
                        x, y, w, h = boxes_mask[i]
                        label = str(classes_mask[class_ids_mask[i]])
                        confidence_mask = confidences_mask[i]
                        color = colors[class_ids_mask[i]]
                        cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
                        cv2.rectangle(frame2, (x, y), (x + w, y + 30), color, -1)
                        cv2.putText(frame2, label + " " + str(round(confidence_mask, 2)), (x, y + 30), font, 3, (255,255,255), 3)
                        
                        




            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    if (class_ids[i]==0):
                        confidence = confidences[i]
             
                        mid_1=int((2*x+w)/2)
                        mid_2=int((2*y+2*h)/2)
                        cv2.circle(frame2,(mid_1,mid_2), 10, (0,0,255),-1)
                        pt_2=np.matmul(projective_matrix_2,[mid_1,mid_2,1])
                        
                        #print(pt_1/pt_1[2])
                        pt_2=pt_2/pt_2[2]
                        #cv2.circle(summation,(int(pt_2[0]),int(pt_2[1])), 10, (0,23,255),-1)
                        points.append((int(pt_2[0]),int(pt_2[1])))
                        cv2.circle(frame2,(mid_1,mid_2), 10, (0,0,255),-1)



            elapsed_time = time.time() - starting_time
            fps = (frame_id_2 / elapsed_time)
            cv2.putText(frame2, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
                    

            
            frame_id_3 += 1
            # Detecting objects
            blob_3 = cv2.dnn.blobFromImage(frame3, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob_3)
            net_mask.setInput(blob_3)
            outs_mask=net_mask.forward(output_layers_mask)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7:
                        # Object detected
                        center_x = int(detection[0] * width3)
                        center_y = int(detection[1] * height3)
                        w = int(detection[2] * width3)
                        h = int(detection[3] * height3)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
               # Showing informations on the screen for mask
            class_ids_mask = []
            confidences_mask = []
            boxes_mask = []
            for out_mask in outs_mask:
                for detection_mask in out_mask:
                    scores_mask = detection_mask[5:]
                    class_id_mask = np.argmax(scores_mask)
                    confidence_mask = scores_mask[class_id_mask]
                    if confidence_mask > 0.1:
                        # Object detected
                        center_x = int(detection_mask[0] * width3)
                        center_y = int(detection_mask[1] * height3)
                        w = int(detection_mask[2] * width3)
                        h = int(detection_mask[3] * height3)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes_mask.append([x, y, w, h])
                        confidences_mask.append(float(confidence_mask))
                        class_ids_mask.append(class_id_mask)

            indexes_mask = cv2.dnn.NMSBoxes(boxes_mask, confidences_mask, 0.4, 0.3)

            for i in range(len(boxes_mask)):
                if i in indexes_mask:
                    x, y, w, h = boxes_mask[i]
                    label = str(classes_mask[class_ids_mask[i]])
                    
                    x, y, w, h = boxes_mask[i]
                    label = str(classes_mask[class_ids_mask[i]])
                    confidence_mask = confidences_mask[i]
                    color = colors[class_ids_mask[i]]
                    cv2.rectangle(frame3, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(frame3, (x, y), (x + w, y + 30), color, -1)
                    cv2.putText(frame3, label + " " + str(round(confidence_mask, 2)), (x, y + 30), font, 3, (255,255,255), 3)
                    
                    

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    if (class_ids[i]==0):
                        confidence = confidences[i]
           
                        #print(center_x,center_y)
                        mid_1=int((2*x+w)/2)
                        mid_2=int((2*y+2*h)/2)
                        cv2.circle(frame3,(mid_1,mid_2), 10, (0,0,255),-1)
                        pt_3=np.matmul(projective_matrix_3,[mid_1,mid_2,1])
                        
                        #print(pt_1/pt_1[2])
                        pt_3=pt_3/pt_3[2]
                        #cv2.circle(summation,(int(pt_3[0]),int(pt_3[1])), 10, (0,23,255),-1)
                        cv2.circle(frame3,(mid_1,mid_2), 10, (0,0,255),-1)
                        points.append((int(pt_3[0]),int(pt_3[1])))





            elapsed_time = time.time() - starting_time
            fps = (frame_id_3 / elapsed_time)
            cv2.putText(frame3, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)

               
           
           
 



            #print((pt_1[0],pt_1[1]))
            #cv2.circle(summation,(int(pt_1[0]),int(pt_1[1])), 10, (0,23,255),-1)
            cv2.imshow('Final Result ', summation)

            cv2.imshow(windowName1, frame1)
            cv2.imshow(windowName2, frame2)
            cv2.imshow(windowName3, frame3)
            offline_data[0] = points.copy()
            offline_heatmap_points[0].append(points)
            final = heatmap(sum(offline_heatmap_points[0][-10:], []), top_view.copy()/255)
            frame_points.append(points)
            some = sum(frame_points, [])
            fin = heatmap(some, top_view/255)
            

          
            cv2.imshow('AnimatedHeatmap', cv2.cvtColor((final).astype(np.float32), cv2.COLOR_RGB2BGR))
            cv2.imshow('StaticHeatmap', cv2.cvtColor((fin).astype(np.float32), cv2.COLOR_RGB2BGR))


           

            # saves the frame from camera 1
            optputFile1.write(frame1)
            optputFile2.write(frame2)
            optputFile3.write(frame3)

            # escape key (27) to exit
            if cv2.waitKey(1) == 27:
                break
        capture1.release()
        cv2.destroyAllWindows()

        
main()
   
