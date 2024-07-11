import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
from easygui import *
import os
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string
import moviepy.editor as mp1
from src.backbone import TFLiteModel, get_model
from src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates, load_json_file 
from src.config import SEQ_LEN, THRESH_HOLD
import numpy as np
import cv2
import time
import mediapipe as mp

# obtain audio from the microphone
def func():
    r = sr.Recognizer()
    isl_gif = [
        "any questions",
        "are you angry",
        "are you busy",
        "are you hungry",
        "are you sick",
        "be careful",
        "can we meet tomorrow",
        "did you book tickets",
        "did you finish homework",
        "do you go to office",
        "do you have money",
        "do you want something to drink",
        "do you want tea or coffee",
        "do you watch TV",
        "dont worry",
        "flower is beautiful",
        "good afternoon",
        "good evening",
        "good morning",
        "good night",
        "good question",
        "had your lunch",
        "happy journey",
        "hello what is your name",
        "how many people are there in your family",
        "i am a clerk",
        "i am bore doing nothing",
        "i am fine",
        "i am sorry",
        "i am thinking",
        "i am tired",
        "i dont understand anything",
        "i go to a theatre",
        "i love to shop",
        "i had to say something but i forgot",
        "i have headache",
        "i like pink colour",
        "i live in nagpur",
        "lets go for lunch",
        "my mother is a homemaker",
        "my name is john",
        "nice to meet you",
        "no smoking please",
        "open the door",
        "please call me later",
        "please clean the room",
        "please give me your pen",
        "please use dustbin dont throw garbage",
        "please wait for sometime",
        "shall I help you",
        "shall we go together tommorow",
        "sign language interpreter",
        "sit down",
        "stand up",
        "take care",
        "there was traffic jam",
        "wait I am thinking",
        "what are you doing",
        "what is the problem",
        "what is todays date",
        "what is your father do",
        "what is your job",
        "what is your mobile number",
        "what is your name",
        "whats up",
        "when is your interview",
        "when we will go",
        "where do you stay",
        "where is the bathroom",
        "where is the police station",
        "you are wrong",
        "address",
        "agra",
        "ahemdabad",
        "all",
        "april",
        "assam",
        "august",
        "australia",
        "badoda",
        "banana",
        "banaras",
        "banglore",
        "bihar",
        "bihar",
        "bridge",
        "cat",
        "chandigarh",
        "chennai",
        "christmas",
        "church",
        "clinic",
        "coconut",
        "crocodile",
        "dasara",
        "deaf",
        "december",
        "deer",
        "delhi",
        "dollar",
        "duck",
        "febuary",
        "friday",
        "fruits",
        "glass",
        "grapes",
        "gujrat",
        "hello",
        "hindu",
        "hyderabad",
        "india",
        "january",
        "jesus",
        "job",
        "july",
        "july",
        "karnataka",
        "kerala",
        "krishna",
        "litre",
        "mango",
        "may",
        "mile",
        "monday",
        "mumbai",
        "museum",
        "muslim",
        "nagpur",
        "october",
        "orange",
        "pakistan",
        "pass",
        "police station",
        "post office",
        "pune",
        "punjab",
        "rajasthan",
        "ram",
        "restaurant",
        "saturday",
        "september",
        "shop",
        "sleep",
        "southafrica",
        "story",
        "sunday",
        "tamil nadu",
        "temperature",
        "temple",
        "thursday",
        "toilet",
        "tomato",
        "town",
        "tuesday",
        "usa",
        "village",
        "voice",
        "wednesday",
        "weight",
        "please wait for sometime",
        "what is your mobile number",
        "what are you doing",
        "are you busy",
    ]

    arr = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    with sr.Microphone() as source:
        # image   = "signlang.png"
        # msg="HEARING IMPAIRMENT ASSISTANT"
        # choices = ["Live Voice","All Done!"]
        # reply   = buttonbox(msg,image=image,choices=choices)
        r.adjust_for_ambient_noise(source)
        i = 0
        while True:
            print("I am Listening")
            audio = r.listen(source)
            # recognize speech using Sphinx
            try:
                a = r.recognize_google(audio)
                a = a.lower()
                print("You Said: " + a.lower())

                for c in string.punctuation:
                    a = a.replace(c, "")

                if (
                    a.lower() == "goodbye"
                    or a.lower() == "good bye"
                    or a.lower() == "bye"
                ):
                    print("oops!Time To say good bye")
                    break

                elif a.lower() in isl_gif:

                    class ImageLabel(tk.Label):
                        """a label that displays images, and plays them if they are gifs"""

                        def load(self, im):
                            if isinstance(im, str):
                                im = Image.open(im)
                            self.loc = 0
                            self.frames = []

                            try:
                                for i in count(1):
                                    self.frames.append(ImageTk.PhotoImage(im.copy()))
                                    im.seek(i)
                            except EOFError:
                                pass

                            try:
                                self.delay = im.info["duration"]
                            except:
                                self.delay = 100

                            if len(self.frames) == 1:
                                self.config(image=self.frames[0])
                            else:
                                self.next_frame()

                        def unload(self):
                            self.config(image=None)
                            self.frames = None

                        def next_frame(self):
                            if self.frames:
                                self.loc += 1
                                self.loc %= len(self.frames)
                                self.config(image=self.frames[self.loc])
                                self.after(self.delay, self.next_frame)

                    root = tk.Tk()
                    lbl = ImageLabel(root)
                    lbl.pack()
                    lbl.load(r"ISL_Gifs/{0}.gif".format(a.lower()))
                    root.mainloop()
                else:
                    for i in range(len(a)):
                        if a[i] in arr:

                            ImageAddress = "letters/" + a[i] + ".jpg"
                            ImageItself = Image.open(ImageAddress)
                            ImageNumpyFormat = np.asarray(ImageItself)
                            plt.imshow(ImageNumpyFormat)
                            plt.draw()
                            plt.pause(0.8)
                        else:
                            continue
            except:
                print(" ")
            plt.close()


# ----------------------------------------------------------------
def vid(video_path):
    r = sr.Recognizer()
    isl_gif = [
        "any questions",
        "are you angry",
        "are you busy",
        "are you hungry",
        "are you sick",
        "be careful",
        "can we meet tomorrow",
        "did you book tickets",
        "did you finish homework",
        "do you go to office",
        "do you have money",
        "do you want something to drink",
        "do you want tea or coffee",
        "do you watch TV",
        "dont worry",
        "flower is beautiful",
        "good afternoon",
        "good evening",
        "good morning",
        "good night",
        "good question",
        "had your lunch",
        "happy journey",
        "hello what is your name",
        "how many people are there in your family",
        "i am a clerk",
        "i am bore doing nothing",
        "i am fine",
        "i am sorry",
        "i am thinking",
        "i am tired",
        "i dont understand anything",
        "i go to a theatre",
        "i love to shop",
        "i had to say something but i forgot",
        "i have headache",
        "i like pink colour",
        "i live in nagpur",
        "lets go for lunch",
        "my mother is a homemaker",
        "my name is john",
        "nice to meet you",
        "no smoking please",
        "open the door",
        "please call me later",
        "please clean the room",
        "please give me your pen",
        "please use dustbin dont throw garbage",
        "please wait for sometime",
        "shall I help you",
        "shall we go together tommorow",
        "sign language interpreter",
        "sit down",
        "stand up",
        "take care",
        "there was traffic jam",
        "wait I am thinking",
        "what are you doing",
        "what is the problem",
        "what is todays date",
        "what is your father do",
        "what is your job",
        "what is your mobile number",
        "what is your name",
        "whats up",
        "when is your interview",
        "when we will go",
        "where do you stay",
        "where is the bathroom",
        "where is the police station",
        "you are wrong",
        "address",
        "agra",
        "ahemdabad",
        "all",
        "april",
        "assam",
        "august",
        "australia",
        "badoda",
        "banana",
        "banaras",
        "banglore",
        "bihar",
        "bihar",
        "bridge",
        "cat",
        "chandigarh",
        "chennai",
        "christmas",
        "church",
        "clinic",
        "coconut",
        "crocodile",
        "dasara",
        "deaf",
        "december",
        "deer",
        "delhi",
        "dollar",
        "duck",
        "febuary",
        "friday",
        "fruits",
        "glass",
        "grapes",
        "gujrat",
        "hello",
        "hindu",
        "hyderabad",
        "india",
        "january",
        "jesus",
        "job",
        "july",
        "july",
        "karnataka",
        "kerala",
        "krishna",
        "litre",
        "mango",
        "may",
        "mile",
        "monday",
        "mumbai",
        "museum",
        "muslim",
        "nagpur",
        "october",
        "orange",
        "pakistan",
        "pass",
        "police station",
        "post office",
        "pune",
        "punjab",
        "rajasthan",
        "ram",
        "restaurant",
        "saturday",
        "september",
        "shop",
        "sleep",
        "southafrica",
        "story",
        "sunday",
        "tamil nadu",
        "temperature",
        "temple",
        "thursday",
        "toilet",
        "tomato",
        "town",
        "tuesday",
        "usa",
        "village",
        "voice",
        "wednesday",
        "weight",
        "please wait for sometime",
        "what is your mobile number",
        "what are you doing",
        "are you busy",
    ]

    arr = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]

    video = mp1.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile("temp_audio.wav")

    with sr.AudioFile("temp_audio.wav") as source:
        r.adjust_for_ambient_noise(source)
        i = 0
        for i in range(1):
            print("I am Listening")
            audio = r.listen(source)

            try:
                a = r.recognize_google(audio)
                a = a.lower()
                print("You Said: " + a.lower())

                for c in string.punctuation:
                    a = a.replace(c, "")

                if (
                    a.lower() == "goodbye"
                    or a.lower() == "good bye"
                    or a.lower() == "bye"
                ):
                    print("oops!Time To say good bye")
                    break

                elif a.lower() in isl_gif:

                    class ImageLabel(tk.Label):
                        """a label that displays images, and plays them if they are gifs"""

                        def load(self, im):
                            if isinstance(im, str):
                                im = Image.open(im)
                            self.loc = 0
                            self.frames = []

                            try:
                                for i in count(1):
                                    self.frames.append(ImageTk.PhotoImage(im.copy()))
                                    im.seek(i)
                            except EOFError:
                                pass

                            try:
                                self.delay = im.info["duration"]
                            except:
                                self.delay = 100

                            if len(self.frames) == 1:
                                self.config(image=self.frames[0])
                            else:
                                self.next_frame()

                        def unload(self):
                            self.config(image=None)
                            self.frames = None

                        def next_frame(self):
                            if self.frames:
                                self.loc += 1
                                self.loc %= len(self.frames)
                                self.config(image=self.frames[self.loc])
                                self.after(self.delay, self.next_frame)

                    root = tk.Tk()
                    lbl = ImageLabel(root)
                    lbl.pack()
                    lbl.load(r"ISL_Gifs/{0}.gif".format(a.lower()))
                    root.mainloop()
                else:
                    for i in range(len(a)):
                        if a[i] in arr:

                            ImageAddress = "letters/" + a[i] + ".jpg"
                            ImageItself = Image.open(ImageAddress)
                            ImageNumpyFormat = np.asarray(ImageItself)
                            plt.imshow(ImageNumpyFormat)
                            plt.draw()
                            plt.pause(0.8)
                        else:
                            continue
            except:
                print(" ")
            plt.close()


# ----------------------------------------------------------------
def sign():
    mp_holistic = mp.solutions.holistic 
    mp_drawing = mp.solutions.drawing_utils

    s2p_map = {k.lower():v for k,v in load_json_file("src/sign_to_prediction_index_map.json").items()}
    p2s_map = {v:k for k,v in load_json_file("src/sign_to_prediction_index_map.json").items()}
    encoder = lambda x: s2p_map.get(x.lower())
    decoder = lambda x: p2s_map.get(x)

    models_path = [
                    './models/islr-fp16-192-8-seed_all42-foldall-last.h5',
    ]
    models = [get_model() for _ in models_path]

    # Load weights from the weights file.
    for model,path in zip(models,models_path):
        model.load_weights(path)

    def real_time_asl():
        """
        Perform real-time ASL recognition using webcam feed.

        This function initializes the required objects and variables, captures frames from the webcam, processes them for hand tracking and landmark extraction, and performs ASL recognition on a sequence of landmarks.

        Args:
            None

        Returns:
            None
        """
        res = []
        tflite_keras_model = TFLiteModel(islr_models=models)
        sequence_data = []
        cap = cv2.VideoCapture(0)
        
        start = time.time()
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # The main loop for the mediapipe detection.
            while cap.isOpened():
                ret, frame = cap.read()
                
                start = time.time()
                
                image, results = mediapipe_detection(frame, holistic)
                draw(image, results)
                
                try:
                    landmarks = extract_coordinates(results)
                except:
                    landmarks = np.zeros((468 + 21 + 33 + 21, 3))
                sequence_data.append(landmarks)
                
                sign = ""
                
                # Generate the prediction for the given sequence data.
                if len(sequence_data) % SEQ_LEN == 0:
                    prediction = tflite_keras_model(np.array(sequence_data, dtype = np.float32))["outputs"]

                    if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
                        sign = np.argmax(prediction.numpy(), axis=-1)
                    
                    sequence_data = []
                
                image = cv2.flip(image, 1)
                
                cv2.putText(image, f"{len(sequence_data)}", (3, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                image = cv2.flip(image, 1)
                
                # Insert the sign in the result set if sign is not empty.
                if sign != "" and decoder(sign) not in res:
                    res.insert(0, decoder(sign))
                
                # Get the height and width of the image
                height, width = image.shape[0], image.shape[1]

                # Create a white column
                white_column = np.ones((height // 8, width, 3), dtype='uint8') * 255

                # Flip the image vertically
                image = cv2.flip(image, 1)
                
                # Concatenate the white column to the image
                image = np.concatenate((white_column, image), axis=0)
                
                cv2.putText(image, f"{', '.join(str(x) for x in res)}", (3, 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
                                
                cv2.imshow('Webcam Feed',image)
                
                # Wait for a key to be pressed.
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

    real_time_asl()
    

while True:
    image = "signlang.png"
    msg = "HEARING IMPAIRMENT ASSISTANT"
    choices = ["video ","Sign", "Live Voice", "All Done!"]
    reply = buttonbox(msg, image=image, choices=choices)
    if reply == choices[0]:
        video_path = enterbox(msg="Please enter the path of the video file:")
        if video_path:
            vid(video_path)
        else:
            print("Invalid video path. Please try again.")
    if reply == choices[1]:
        sign()
    if reply == choices[2]:
        func()
    if reply == choices[3]:
        quit()
