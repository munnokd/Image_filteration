import datetime
import os
import tkinter
from tkinter import messagebox
import numpy as np
import imutils
import cv2
from tkinter import filedialog
from tkinter import *
from tkinter.messagebox import askyesno
from tkinter.ttk import Style
from PIL import ImageEnhance
from functools import partial
from PIL.ImageEnhance import Contrast
from PIL import *
from matplotlib import image
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from datetime import datetime

global root
root = Tk()
root.title("Photo Editor")
root.geometry('1500x1500')
root.configure(bg='lightblue')


def ShowAllButton(path, canvas): #show all button
    back = Button(root, text="Back", command=lambda: showImage(path, canvas), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    back.place(x=20, y=100)
    edgedetection = Button(root, text="Edge Detection", command=lambda: EdgeDetection(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    edgedetection.place(x=20, y=200)
    blur = Button(root, text="Blur", command=lambda: Blur(canvas, path), height=3, width=15, borderwidth=3, activebackground='#345',activeforeground='lightgreen')
    blur.place(x=20, y=300)
    cartoonize = Button(root, text="Cartoonize", command=lambda: Cartoonize(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightyellow')
    cartoonize.place(x=20, y=400)
    blackandwhite = Button(root, text="Black and White", command=lambda: BlackandWhite(canvas, path), height=3,width=15, borderwidth=3, activebackground='white', activeforeground='lightgreen')
    blackandwhite.place(x=20, y=500)
    enhance = Button(root, text="Enhance", command=lambda: Enhancement(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    enhance.place(x=1150, y=100)
    bright = Button(root, text="Brightness", command=lambda: Brightness(path, canvas), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    bright.place(x=1150, y=200)
    rotate = Button(root, text="Rotate", command=lambda: Rotate(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    rotate.place(x=1150, y=300)
    text = Button(root, text="Add Text", command=lambda: takeText(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    text.place(x=1150, y=400)
    Contrast = Button(root, text="Contrast", command=lambda: contrast(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    Contrast.place(x=1150, y=500)

    histogram = Button(root, text="Histogram", command=lambda: Histogram(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    histogram.place(x=600, y=600)


global last_value_of_con
last_value_of_con = 0


def contrast_change(x, path): # contrast change function
    global last_value_of_con
    img = cv2.imread(path)
    contrast_img = cv2.addWeighted( 
        img, x, np.zeros(img.shape, img.dtype), 0, 0)
    cv2.imshow("Contrast", contrast_img)
    last_value_of_con = x


def Histogram(canvas, path): #histogram function
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    h = np.histogram(img, 255)
    plt.plot(h[0])
    plt.title('histogram')
    plt.show()


def contrast(canvas, path): #contrast function
    trackbars_img = np.uint8(np.full((50, 500, 3), 255))
    cv2.imshow('Contrast', trackbars_img)
    cv2.createTrackbar('Contrast', 'Contrast', 1, 10,partial(contrast_change, path=path))
    cv2.waitKey(0)
    img = cv2.imread(path)
    contrast_img = cv2.addWeighted(
        img, last_value_of_con, np.zeros(img.shape, img.dtype), 0, 0)
    cv2.imwrite('Contrast.png', contrast_img)
    showImage('Contrast.png', canvas)


def take(canvas, path, inputtxt, add): #add text function
    txt = inputtxt.get("1.0", "end-1c")
    inputtxt.destroy()
    add.destroy()
    img = cv2.imread(path)
    new_img = cv2.putText(img, txt, org=(
        0, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 0), thickness=2)
    cv2.imwrite('TextImage.png', new_img)
    showImage('TextImage.png', canvas)


def takeText(canvas, path): #Take Input text for image function
    inputtxt = tkinter.Text(root, height=5, width=20)
    inputtxt.pack()
    add = Button(root, text="Done", command=lambda: take(canvas, path, inputtxt, add), height=3, width=10,activebackground='#345', activeforeground='red')
    add.place(x=400, y=200)


def BlackandWhite(canvas, path): #Black and White function
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    cv2.imwrite('BlackandWhite.png', gray)
    showImage('BlackandWhite.png', canvas)


def showImage(path, canvas):# show image function
    img = PhotoImage(file=path)
    canvas.create_image(0, 0, anchor=NW, image=img)
    root.mainloop()


def Cartoonize(canvas, path): #cartoonize function
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)# Adaptive thresholding typically takes a grayscale or color image as input and, in the simplest implementation, outputs a binary image representing the segmentation.
    color = cv2.bilateralFilter(img, 7, 200, 200) #A bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter for images. 
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    data = np.float32(img).reshape((-1, 3))
    cv2.imwrite('Cartoon.png', cartoon)
    showImage('Cartoon.png', canvas)

global last_value_of_rot
last_value_of_rot = 0


def rotate_change(x, path): #rotate change function
    global last_value_of_rot
    img = cv2.imread(path)
    Rotated_image = imutils.rotate(img, angle=x)
    cv2.imshow("Rotate", Rotated_image)
    last_value_of_rot = x


def Rotate(canvas, path): #rotate function
    trackbars_img = np.uint8(np.full((50, 500, 3), 255))
    cv2.imshow('Rotate', trackbars_img)
    cv2.createTrackbar('Rotate', 'Rotate', 1, 360,partial(rotate_change, path=path))
    cv2.waitKey(0)
    img = cv2.imread(path)
    Rotated_image = imutils.rotate(img, angle=last_value_of_rot)
    cv2.imwrite('Rotate.png', Rotated_image)
    showImage('Rotate.png', canvas)


global last_value_of_brig
last_value_of_brig = 0


def brightness_change(x, path): #brightness change function
    global last_value_of_brig
    img = cv2.imread(path)
    in_matrix = np.ones(img.shape, dtype='uint8') * x
    new_img = cv2.add(img, in_matrix)
    cv2.imshow("Bright", new_img)
    last_value_of_brig = x


def Brightness(path, canvas): #brightness function
    trackbars_img = np.uint8(np.full((50, 500, 3), 255))
    cv2.imshow('Brightness', trackbars_img)
    cv2.createTrackbar('Brightness', 'Brightness', 50, 100,partial(brightness_change, path=path))
    cv2.waitKey(0)
    img = cv2.imread(path)
    in_matrix = np.ones(img.shape, dtype='uint8') * last_value_of_brig
    new_img = cv2.add(img, in_matrix)
    cv2.imwrite('Bright.png', new_img)
    showImage('Bright.png', canvas)


def Blur(canvas, path): #blur function
    img = cv2.imread(path)
    Gaussian = cv2.GaussianBlur(img, (21, 21), 0)
    cv2.imwrite('Blur.png', Gaussian)
    showImage('Blur.png', canvas)


def EdgeDetection(canvas, path): #edge detection function
    img = cv2.imread(path)
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite('EdgeDetection.png', edges)
    showImage('EdgeDetection.png', canvas)


def bilateral(canvas, path):#Median function
    img = cv2.imread(path)
    img_new = cv2.bilateralFilter(img, 15, 75, 75)
    cv2.imwrite('bilateral.png', img_new)
    showImage('bilateral.png', canvas)


def Avarage(canvas, path):#avarage function
    img = cv2.imread(path)
    img_new = cv2.medianBlur(img, 5)
    cv2.imwrite('Avarage.png', img_new)
    showImage('Avarage.png', canvas)


def Gaussian(canvas, path):#gaussian function
    img = cv2.imread(path)
    img_new = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite('GaussianBlur.png', img_new)
    showImage('GaussianBlur.png', canvas)


def Erosion(canvas, path):#erosion function
    img = cv2.imread(path)
    kernel = np.ones((5, 5), np.uint8)
    img_new = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite('Erosion.png', img_new)
    showImage('Erosion.png', canvas)


def Dilation(canvas, path):#dilation function
    img = cv2.imread(path)
    kernel = np.ones((5, 5), np.uint8)
    img_new = cv2.dilate(img, kernel, iterations=1)
    cv2.imwrite('Dilation.png', img_new)
    showImage('Dilation.png', canvas)


def Lowpass(canvas, path):#lowpass function
    img = cv2.imread(path)
    kernel = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])

    kernel = kernel/(np.sum(kernel) if np.sum(kernel) != 0 else 1)

    # filter the source image
    img_new = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('Lowpass.png', img_new)
    showImage('Lowpass.png', canvas)


def HighPass(canvas, path):#highpass function
    img = cv2.imread(path)
    kernel = np.array([[0.0, -1.0, 0.0],[-1.0, 4.0, -1.0],[0.0, -1.0, 0.0]])

    kernel = kernel/(np.sum(kernel) if np.sum(kernel) != 0 else 1)

    # filter the source image
    img_new = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('HighPass.png', img_new)
    showImage('HighPass.png', canvas)


def gamma_change(x, path):#gamma change function
    global last_value_of_gamma
    img = cv2.imread(path)
    # in_matrix = np.ones(img.shape, dtype='uint8') * x
    # new_img = cv2.add(img, in_matrix)
    new_img = np.array(255*(img / 255) ** x, dtype='uint8')
    cv2.imshow("Gamma", new_img)
    last_value_of_gamma = x


def Gamma(canvas, path):#gamma function
    trackbars_img = np.uint8(np.full((50, 500, 3), 255))
    cv2.imshow('Gamma', trackbars_img)
    cv2.createTrackbar('Gamma', 'Gamma', 0, 8,partial(gamma_change, path=path))

    cv2.setTrackbarMin('Gamma', 'Gamma', -3)
    cv2.waitKey(0)
    img = cv2.imread(path)
    new_img = np.array(255*(img / 255) ** last_value_of_gamma, dtype='uint8')
    cv2.imwrite('Gamma.png', new_img)
    showImage('Gamma.png', canvas)


def Enhancement(canvas, path):#enhancement function
    # img = cv2.imread(path)
    newWindow = Toplevel(root)

    # sets the title of the
    # Toplevel widget
    newWindow.title("Image Enhancement")

    # sets the geometry of toplevel
    newWindow.geometry("600x600")

    # A Label widget to show in toplevel
    Label(newWindow,text="This is a new window").pack()
    Bilateral = Button(newWindow, text="Median", command=lambda: bilateral(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    Bilateral.place(x=20, y=100)
    gamma = Button(newWindow, text="Gamma", command=lambda: Gamma(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    gamma.place(x=20, y=200)

    avarage = Button(newWindow, text="avarage", command=lambda: Avarage(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    avarage.place(x=200, y=100)

    GaussianBlur = Button(newWindow, text="Gaussian", command=lambda: Gaussian(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    GaussianBlur.place(x=200, y=200)

    erosion = Button(newWindow, text="erosion", command=lambda: Erosion(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    erosion.place(x=400, y=100)
    dilation = Button(newWindow, text="dilation", command=lambda: Dilation(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    dilation.place(x=400, y=200)

    lowpass = Button(newWindow, text="lowpass", command=lambda: Lowpass(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    lowpass.place(x=200, y=300)

    highPass = Button(newWindow, text="highPass", command=lambda: HighPass(canvas, path), height=3, width=15, borderwidth=3,activebackground='white', activeforeground='lightgreen')
    highPass.place(x=400, y=300)


    # alpha = 1.5
    # beta = 10
    # adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    # cv2.imwrite('Enhancement.png', adjusted)
    # showImage('Enhancement.png', canvas)
global canvas


def CreateFrame(path):#create frame function
    root.geometry('2000x1500')
    im = Image.open(path)
    canvas = Canvas(root, width=im.width, height=im.height)
    canvas.pack()
    ShowAllButton(path, canvas)
    showImage(path, canvas)


global path


def SelectImage(): # select image
    path = filedialog.askopenfilename()
    if path != "":
        b.pack_forget()  # remove button
        CreateFrame(path) # create frame


global b

b = Button(root, text="Select Image", font=('arial bold', 10), command=SelectImage, height=3, width=10, borderwidth=10,activebackground='white', activeforeground='lightgreen') # Create Button
b.pack(pady=300)

root.mainloop() # Execute Tkinter
