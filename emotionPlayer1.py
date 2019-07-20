import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import vlc
import time

em=['angrey','fear','happy','sad','surprise','neutral']
playList=['/home/manish/Desktop/ML/Day-9/PlayList/Imagine-Dragons-Demons.mp3',
         '/home/manish/Desktop/ML/Day-9/PlayList/Lag Jaa Gale .mp3',
         '/home/manish/Desktop/ML/Day-9/PlayList/Wang - Ammy Virk.mp3',
         '/home/manish/Desktop/ML/Day-9/PlayList/Lukas Graham - 7 Years.mp3',
         '/home/manish/Desktop/ML/Day-9/PlayList/havana.mp3',
         '/home/manish/Desktop/ML/Day-9/PlayList/Eastside .mp3']


def playSong(emotionInd):
    player=vlc.MediaPlayer(playList[emotionInd])
    print(em[emotionInd])
    v=cv2.VideoCapture(0)
    fd=cv2.CascadeClassifier('/home/manish/Desktop/ML/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')
    flag=0
    while True:
        r,i=v.read()
        j=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        f=fd.detectMultiScale(j,1.24,5)
        #print(f,len(f))
        
        if(len(f)>0):
            [x,y,w,h]=f[0]
            cv2.rectangle(i,(x,y),(x+w,y+h),(13,196,255),5)
            player.play()
            player.audio_set_volume(100)
            flag=1
        elif (flag==1):
            player.pause()
            flag=0

        #print(player.get_state())   
        cv2.imshow('Cam',i)
        k=cv2.waitKey(5)
        if(k==ord('s')):
            cv2.destroyAllWindows()
            player.stop()
            v.release()
            break
        elif(k==ord('e')):
            cv2.destroyAllWindows()
            player.stop()
            v.release()
            break
            return(0)

def faceEmotion()
    v=cv2.VideoCapture(0)
    fd=cv2.CascadeClassifier('/home/manish/Desktop/ML/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')
    model=load_model('/home/manish/Desktop/ML/Projects/EmotionPlayer/emotion_recognition.h5',compile=False)
    
    countList=[0,0,0,0,0,0]
    noFrames=0
    while(noFrames!=80):
        print(noFrames)    
        try:
            r,live=v.read()
            j=cv2.cvtColor(live,cv2.COLOR_BGR2GRAY)
            f=fd.detectMultiScale(j,1.25,5)
            #print(f)
            if(len(f)>0):
                [x,y,w,h]=f[0]
                cv2.rectangle(live,(x,y),(x+w,y+h),(0,0,255),5)
                roi=j[x:x+w,y:y+h]
                roi=cv2.resize(roi,(48,48))
                roi=roi.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)
                p=list(model.predict(roi)[0])
                print(em[p.index(max(p))])
                countList[p.index(max(p))]+=1
                noFrames+=1
            cv2.imshow('img',live)
            k=cv2.waitKey(5)
                
                
        except AssertionError as e:
            print('Error'+e)
    cv2.destroyAllWindows()
    v.release()
    time.sleep(1)
    print(countList)
    emotionInd=countList.index(max(countList))
    var=playSong(emotionInd)
    
    
