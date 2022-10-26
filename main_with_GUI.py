import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors, metrics, mixture, svm
from sklearn.model_selection import train_test_split, GridSearchCV

import tkinter as tk
from tkinter import *
window = tk.Tk()

window.title('Prog for Data Science')
# width x height + x_offset + y_offset:
window.geometry("1064x720+100+100")

#Set font
myfont = "Calibri, 14"
myfont1 = "Arial, 20"

#Add a label-heading
lbl_header = tk.Label(text="CLASSIFICATION OF DATASETS USING MACHINE LEARNING",fg="white", bg="darkblue", font=myfont1, height=4)
lbl_header.pack()


# ### Frame1-to select dataset
#Create frame 1 for dataset
frame_1 = LabelFrame(window, text= "Choose Dataset..." , padx=5,pady=5)
frame_1.pack(padx=20, pady=20)

#Add variable var and 3 radio buttons
var = tk.StringVar()
var.set(None)

rb1 = tk.Radiobutton(frame_1, text="Iris Dataset", variable=var, value='i', font=myfont)
rb1.grid(row=2, column=0)
rb1.deselect()

rb2 = tk.Radiobutton(frame_1, text="Breast Cancer Dataset", variable=var, value='b', font=myfont)
rb2.grid(row=3, column=0)
rb2.deselect()

rb3 = tk.Radiobutton(frame_1, text="Wine Dataset", variable=var, value='w', font=myfont)
rb3.grid(row=4, column=0)
rb3.deselect()

#Label to display output when button is clicked
lb_output = tk.Label(text="", fg="navy", font=myfont)

#################################
def select_d():
    selected = var.get()
    output = ''
    if selected == 'i':

        output = 'iris data selected'
    elif selected == 'b':
        output = 'breast cancer data selected'
    elif selected == 'w':
        output = 'wine data selected'
    #lb_output.config(text=output)

    return output
#################################

# ### Frame2 to select classifer

#Create frame 2 for classifier
frame_2 = LabelFrame(window, text= "Choose Classifier..." , padx=5,pady=5)
frame_2.pack(padx=20, pady=20)

#Add variable var and 2 radio buttons
var1 = tk.StringVar()
var1.set(None)

cb1 = tk.Radiobutton(frame_2, text="k-Nearest Neighbour Classifier", variable=var1, value='c1', font=myfont)
cb1.grid(row=0, column=1)
cb1.deselect()

cb2 = tk.Radiobutton(frame_2, text="Support Vector Classifier", variable=var1, value='c2', font=myfont)
cb2.grid(row=1, column=1)
cb2.deselect()

#Label to display output when button is clicked
lb2_output1 = tk.Label(text="", fg="navy", font=myfont)

#################################
def select_c():
    selected = var1.get()
    output1 = ''
    if selected == 'c1':
        output1 = 'kNN classifier'
    else:
        output1 = 'SVM classifier'
    #lb2_output1.config(text=output1)
    return output1
#################################

# ### Frame 3 to select value for k-Fold(3,5,7 or more)

#Create frame 3 for cv folds
frame_3 = LabelFrame(window, text= "Choose k-Folds..." , padx=5,pady=5)
frame_3.pack(padx=20, pady=20)

#Add variable var and 4 radio buttons
var2 = tk.StringVar()
var2.set(None)

rb1 = tk.Radiobutton(frame_3, text="k=3", variable=var2, value='k3', font=myfont)
rb1.grid(row=0, column=1)
rb1.deselect()

rb2 = tk.Radiobutton(frame_3, text="k=5", variable=var2, value='k5', font=myfont)
rb2.grid(row=1, column=1)
rb2.deselect()

rb3 = tk.Radiobutton(frame_3, text="k=7", variable=var2, value='k7', font=myfont)
rb3.grid(row=2, column=1)
rb3.deselect()

rb4 = tk.Radiobutton(frame_3, text="k=10", variable=var2, value='k10', font=myfont)
rb4.grid(row=3, column=1)
rb4.deselect()

#Label to display output when button is clicked
lb3_output2 = tk.Label(text="", fg="navy",font=myfont)


#################################
def select_k():
    selected = var2.get()
    output2 = ''
    if selected == 'k3':
        output2 = 'k=3'
    elif selected == 'k5':
        output2 = 'k=5'
    elif selected == 'k7':
        output2 = 'k=7'
    elif selected == 'k10':
        output2 = 'k=10'

    # lb3_output2.config(text=output2)
    return output2
#################################

# ### Add Run button to start running classifier
from tkinter.messagebox import showinfo

def button_clicked(d,c,k):
    x = d+c+k
    showinfo(title='Information', message= x)

#################################
def run():
    d = select_d()
    c = select_c()
    k = select_k()
    #call some logic
    #call ouput
    button_clicked(d,c,k)
    cv=fn1(d,c,k)
    cm=fn2(d,c,k)
    display_fn1(cv, frame_4)
    display_fn2(cm, frame_5)

#################################
#def fn1

# Add Run button
b4 = tk.Button(text="Run", fg="white", bg="darkblue", width=10, height=1, font=myfont, command=run)
b4.pack(padx=20, pady=20)


window.mainloop()
###output accuracy,etc to txt file