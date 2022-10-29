import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors, metrics, mixture, svm
from sklearn.model_selection import train_test_split, GridSearchCV


import tkinter as tk
from tkinter import *
from sklearn import datasets, neighbors, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
import matplotlib.pyplot as plt
import numpy as np
from warnings import simplefilter


def frame_formatter(window, dimensions):
    window.title('Prog for Data Science')
    # width x height + x_offset + y_offset:
    window.geometry(dimensions)
    # Set font
    myfont = "Calibri, 9"
    myfont1 = "Arial, 12"
    lbl_header = tk.Label(window, text="CLASSIFICATION OF DATASETS USING ML", fg="white", bg="darkblue",
                          font=myfont1)
    # lbl_header.pack()
    lbl_header.grid(row=0, column=0)
    return myfont


def select_d(var):
    selected = var.get()
    output = datasets._base
    if selected == 'i':
        output = datasets.load_iris()  # 'iris data selected'
    elif selected == 'b':
        output = datasets.load_breast_cancer()  # 'breast cancer data selected'
    elif selected == 'w':
        output = datasets.load_wine()  # 'wine data selected'
    # lb_output.config(text=output)
    return output


def frame_one(window, myfont):
    # Create frame 1 for dataset
    frame_1 = LabelFrame(window, text="Choose Dataset...", padx=5, pady=5)
    # frame_1.pack(padx=20, pady=20)
    # frame_1.place(x=20,y=140)
    frame_1.grid(row=3, column=0)
    # Add variable var and 3 radio buttons
    var = tk.StringVar()
    var.set("")

    rb1 = tk.Radiobutton(frame_1, text="Iris Dataset", variable=var, value='i', font=myfont)
    rb1.grid(row=2, column=0)
    rb1.deselect()

    rb2 = tk.Radiobutton(frame_1, text="Breast Cancer Dataset", variable=var, value='b', font=myfont)
    rb2.grid(row=3, column=0)
    rb2.deselect()

    rb3 = tk.Radiobutton(frame_1, text="Wine Dataset", variable=var, value='w', font=myfont)
    rb3.grid(row=4, column=0)
    rb3.deselect()
    return var


def select_c(var1):
    selected = var1.get()
    output1 = ''
    if selected == 'c1':
        output1 = neighbors.KNeighborsClassifier()  # 'kNN classifier'
    elif selected == 'c2':
        output1 = neighbors.KNeighborsClassifier()  # TODO @joy -- 'SVM classifier'
    # lb3_output2.config(text=output2)
    return output1


def frame_two(window, myfont):
    # Create frame 2 for classifier
    frame_2 = LabelFrame(window, text="Choose Classifier...", padx=5, pady=5)
    # frame_2.pack(padx=20, pady=20)
    # frame_2.place(x=20,y=290)
    frame_2.grid(row=4, column=0)
    # Add variable var and 2 radio buttons
    var1 = tk.StringVar()
    var1.set("")

    cb1 = tk.Radiobutton(frame_2, text="k-Nearest Neighbour Classifier", variable=var1, value='c1', font=myfont)
    cb1.grid(row=0, column=0)
    cb1.deselect()

    cb2 = tk.Radiobutton(frame_2, text="Support Vector Classifier", variable=var1, value='c2', font=myfont)
    cb2.grid(row=1, column=0)
    cb2.deselect()
    return var1


def select_k(var2):
    selected = var2.get()
    output2 = [{'n_neighbors': [1, 2, 3]}]
    if selected == 'k3':
        output2 = [{'n_neighbors': [1, 2, 3]}]
    elif selected == 'k5':
        output2 = [{'n_neighbors': [1, 2, 3, 4, 5]}]
    elif selected == 'k7':
        output2 = [{'n_neighbors': [1, 2, 3, 4, 5, 6, 7]}]
    elif selected == 'k10':
        output2 = [{'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
    # lb3_output2.config(text=output2)
    return output2


def frame_three(window, myfont):
    # Create frame 3 for cv folds
    frame_3 = LabelFrame(window, text="Choose k-Folds...", padx=5, pady=5)
    # frame_3.pack(padx=20, pady=20)
    # frame_3.place(x=20,y=400)
    frame_3.grid(row=5, column=0)
    # Add variable var and 4 radio buttons
    var2 = tk.StringVar()
    var2.set("")

    rb1 = tk.Radiobutton(frame_3, text="k=3", variable=var2, value='k3', font=myfont)
    rb1.grid(row=0, column=0)
    rb1.deselect()

    rb2 = tk.Radiobutton(frame_3, text="k=5", variable=var2, value='k5', font=myfont)
    rb2.grid(row=1, column=0)
    rb2.deselect()

    rb3 = tk.Radiobutton(frame_3, text="k=7", variable=var2, value='k7', font=myfont)
    rb3.grid(row=2, column=0)
    rb3.deselect()

    rb4 = tk.Radiobutton(frame_3, text="k=10", variable=var2, value='k10', font=myfont)
    rb4.grid(row=3, column=0)
    rb4.deselect()
    return var2


def fn_cv_score(dataset, classifier, parameter):
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # Loading a dataset
    X = dataset.data
    y = dataset.target
    class_names = dataset.target_names

    # Plot CV SCORE vs parameter
    # Setting the range for the parameter (from 1 to 10)
    #parameter_range = np.arange(1, 10, 1)
    parameter_range = np.array(parameter[0].get('n_neighbors'))

    # Calculate accuracy on training and test set using the
    # gamma parameter with 5-fold cross validation
    train_score, test_score = validation_curve(classifier, X, y, cv=5, param_name='n_neighbors',
                                               param_range=parameter_range, scoring='accuracy')
    # test_score = validation_curve(gscv_classifier, X_test, y_test,cv = 5, param_name = 'estimator__n_neighbors' ,param_range = parameter_range)

    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis=1)
    std_train_score = np.std(train_score, axis=1)

    # Plot mean accuracy scores for training scores
    train_dash_line_high = mean_train_score + std_train_score
    train_dash_line_low = mean_train_score - std_train_score
    # TODO: has to be used somewhare -- std_train_score ; std_test_score
    plt.plot(parameter_range, mean_train_score, label="Training Score", color='b')
    plt.plot(parameter_range, train_dash_line_high, color='b', linestyle='dashed')
    plt.plot(parameter_range, train_dash_line_low, color='b', linestyle='dashed')

    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis=1)
    std_test_score = np.std(test_score, axis=1)

    plt.plot(parameter_range, mean_test_score, label="Cross Validation Score", color='g')

    test_dash_line_high = mean_test_score + std_test_score
    test_dash_line_low = mean_test_score - std_test_score

    plt.plot(parameter_range, test_dash_line_high, color='g', linestyle='dashed')
    plt.plot(parameter_range, test_dash_line_low, color='g', linestyle='dashed')
    # plt.plot(parameter_range, std_train_score,
    #    label = "std_train_score", color = 'r')
    # plt.plot(parameter_range, std_test_score,
    #    label = "std_test_score", color = 'y')

    # Creating the plot
    plt.title("Validation Curve with KNN Classifier")
    plt.xlabel("Number of Neighbours")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
    return 0


# TODO by joy - check if classifier, parameter is used properly as wanted
def fn_plot_confusion_matrix(dataset, classifier, parameter):
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # Loading a dataset
    X = dataset.data
    y = dataset.target
    class_names = dataset.target_names

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    # load grid search cross validation function for parameter estimation
    gscv_classifier = GridSearchCV(
        estimator=classifier,
        param_grid=parameter,
        cv=5,  # 5-fold cross validation
        scoring='accuracy'
    )

    # This function returns classifier gscv_classifier and we use it to train the training set

    gscv_classifier.fit(X_train, y_train)

    # This function returns classifier gscv_classifier and we use it to train the training set

    gscv_classifier.fit(X_train, y_train)

    # Get parameter values, scores (accuracies) and best parameter from this gscv_classifier classifier
    #TODO: make sesible division of print msg
    print("---------------------------------WTF!!! fn_plot_confusion_matrix ----------------------------")
    print("Grid scores on validation set:")
    means = gscv_classifier.cv_results_['mean_test_score']
    stds = gscv_classifier.cv_results_['std_test_score']
    results = gscv_classifier.cv_results_['params']
    for mean, std, param in zip(means, stds, results):
        print("Parameter: %r, accuracy: %0.3f (+/-%0.03f)" % (param, mean, std * 2))
    print()

    print("Best parameter:", gscv_classifier.best_params_)

    # This gscv_classifier classifier now applies the best parameter, so we just use it to test the testing dataset

    y_pred = gscv_classifier.predict(X_test)

    # Plot confusion matrix and accuracy

    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    plotcm = metrics.plot_confusion_matrix(gscv_classifier, X_test, y_test, display_labels=class_names)
    plotcm.ax_.set_title('Accuracy = {0:.2f}%'.format(accuracy))
    plt.show()


def run(var, var1, var2):
    # init dataset, classifier, parameter
    dataset = select_d(var)
    classifier = select_c(var1)
    parameter = select_k(var2)

    # call some logic
    fn_cv_score(dataset, classifier, parameter)
    fn_plot_confusion_matrix(dataset, classifier, parameter)

    # call output
    # button_clicked(dataset, classifier, parameter)
    # xx = display_fn1(cv, "frame4")
    # yy = display_fn2(cm, "frame5")
    # print(xx)
    # print(yy)


def main():
    print("Proj for Data Science...")
    # Header
    window = tk.Tk()
    dimensions = "400x700+100+100"
    myfont = frame_formatter(window, dimensions)

    # display each frames
    dataset = frame_one(window, myfont)
    classifier = frame_two(window, myfont)
    parameter = frame_three(window, myfont)

    # Add Run button
    run_button = tk.Button(window, text="Run", fg="white", bg="darkblue", width=10, height=1, font=myfont,
                           command=lambda: run(dataset, classifier, parameter))
    # run_button.pack(padx=20, pady=20)
    # run_button.place(x=20, y=580)
    run_button.grid(row=6, column=0)

    # Add exit button
    exit_button = Button(window, text="Exit", fg="white", bg="darkblue", width=10, height=1, font=myfont,
                         command=window.destroy)
    # exit_button.pack(padx=20, pady=20)
    # exit_button.place(x=150, y=580)
    exit_button.grid(row=7, column=0)

    # dummy frame 4
    # four_button = tk.Button(text="F4", fg="white", bg="darkblue", width=10, height=1, font=myfont)
    # four_button.place(x=300, y=580)
    # four_button.grid(row=3, column=1, rowspan=5)
    # dummy frame 5
    # five_button = tk.Button(text="F5", fg="white", bg="darkblue", width=10, height=1, font=myfont)
    # five_button.place(x=650, y=580)
    # five_button.grid(row=3, column=2, rowspan=5)

    # loop again for new inputs / persistence
    window.mainloop()


if __name__ == "__main__":
    main()