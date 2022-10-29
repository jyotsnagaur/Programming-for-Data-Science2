import time
import tkinter as tk
from tkinter import *
from sklearn import datasets, neighbors, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
import matplotlib.pyplot as plt
import numpy as np
from warnings import simplefilter


# 2 Function for creating the base window containing title and fonts
def frame_formatter(window, dimensions):
    window.title('Prog for Data Science')
    # width x height + x_offset + y_offset:
    window.geometry(dimensions)
    # Set font
    myfont = "Arial, 9"
    myfont1 = "Arial, 12"
    lbl_header = tk.Label(window, text="CLASSIFICATION OF DATASETS USING ML", fg="white", bg="darkblue",
                          font=myfont1)
    # lbl_header.pack()
    lbl_header.grid(row=0, column=0)
    return myfont


# 3 Function for selecting the dataset.
'''The input is the variable value taken by the radio buttons defined in next step.
The function outputs the loaded dataset'''


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


# 4 Function for creating frame 1 with radio buttons for selecting the dataset.
'''The radio buttons for selecting dataset are contained in frame 1.
The function outputs the variable with value i,b,w for the above function for loading dataset'''


def frame_one(window, myfont):
    # Create frame 1 for dataset
    frame_1 = LabelFrame(window, text="Choose Dataset...", padx=5, pady=5)
    # frame_1.pack(padx=20, pady=20)
    # frame_1.place(x=20,y=140)
    frame_1.grid(row=3, column=0)
    # Add variable var and 3 radio buttons
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
    return var


# 5 Function for selecting the classifier.
'''The input is the variable value taken by the radio buttons defined in next step.
The function outputs the classifier as output1'''


def select_c(var1):
    return var1.get()


def get_classifier(selected):
    output1 = ''
    if selected == 'c1':
        output1 = neighbors.KNeighborsClassifier()  # 'kNN classifier'
    elif selected == 'c2':
        output1 = svm.SVC()
    return output1


# 6 Function for creating frame 2 with radio buttons for selecting the classifier.
'''The radio buttons for selecting classifier are contained in frame 2.
 The function outputs the variable with value c1,c2 for the above function for giving classifier as output'''


def frame_two(window, myfont):
    # Create frame 2 for classifier
    frame_2 = LabelFrame(window, text="Choose Classifier...", padx=5, pady=5)
    # frame_2.pack(padx=20, pady=20)
    # frame_2.place(x=20,y=290)
    frame_2.grid(row=4, column=0)
    # Add variable var and 2 radio buttons
    var1 = tk.StringVar()
    var1.set(None)

    cb1 = tk.Radiobutton(frame_2, text="k-Nearest Neighbour Classifier", variable=var1, value='c1', font=myfont)
    cb1.grid(row=0, column=0)
    cb1.deselect()

    cb2 = tk.Radiobutton(frame_2, text="Support Vector Classifier", variable=var1, value='c2', font=myfont)
    cb2.grid(row=1, column=0)
    cb2.deselect()
    return var1


# 7 Function for selecting the k value for cross validation.
'''The input is the variable value taken by the radio buttons defined in next step.
 The function outputs the cv value used in GridsearchCV'''


def select_k(var2):
    selected = var2.get()
    output2 = ''
    if selected == 'k3':
        output2 = 3
    elif selected == 'k5':
        output2 = 5
    elif selected == 'k7':
        output2 = 7
    elif selected == 'k10':
        output2 = 10
    # lb3_output2.config(text=output2)
    return output2


# 8 Function for creating frame 3 with radio buttons for selecting the number of folds for cross validation.
'''The radio buttons for selecting k-folds are contained in frame 3.
 The function outputs the variable with value k3,k5,k7,k10 for the above function for giving cv as output'''


def frame_three(window, myfont):
    # Create frame 3 for cv folds
    frame_3 = LabelFrame(window, text="Choose k-Folds...", padx=5, pady=5)
    # frame_3.pack(padx=20, pady=20)
    # frame_3.place(x=20,y=400)
    frame_3.grid(row=5, column=0)
    # Add variable var and 4 radio buttons
    var2 = tk.StringVar()
    var2.set(None)

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


# 9 Function for plotting the cv score/accuracy vs parameter(which is n_neighbours for k-nearest neighbour
# and C for SVM).
'''The function takes the dataset,classifier and parameter as input.
 The function outputs the graph between accuracy/cv score and parameter using matplotlib imported as plt in step 1'''


def fn_cv_score(dataset, classifier, fold):
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # Loading a dataset
    X = dataset.data
    y = dataset.target
    # class_names = dataset.target_names # TODO:remove if everything ok

    # Plot CV SCORE vs parameter
    # Setting the range for the parameter
    obj_classifier = get_classifier(classifier)

    if classifier == 'c1':
        parameter_range = np.arange(1, 10, 1)  # setting the range for n_neighbours on x axis
    elif classifier == 'c2':
        parameter_range = np.arange(0.1, 10, 0.4)  # Setting the range for the parameter (from 0.000000001 to 0.1)

    # Setting the parameter name
    if classifier == 'c1':
        p_name = 'n_neighbors'
    elif classifier == 'c2':
        p_name = 'C'

    # Calculate accuracy on training and test set using the
    # selected parameter with k-fold cross validation
    train_score, test_score = validation_curve(obj_classifier, X, y, cv=fold, param_name=p_name,
                                               param_range=parameter_range, scoring='accuracy')

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
    plt.xlabel("Parameter")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
    return 0


def save_as(object_in, filename):
    f = None
    try:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        f = open('output/' + str(filename) + '_' + timestr + '.txt', "w")
        for item in object_in:
            # write each item on a new line
            f.write("%s\n" % str(item))
    except Exception as ex:
        # incase the file is empty
        print(ex.args)
    finally:
        if f:
            f.close()


def my_print(object_in, a):
    object_in.append(a)
    print(a)


# 10 Function for plotting the confusion matrix with accuracy percentage.
'''The function takes the dataset,classifier and parameter as input.
 The function also runs the GridsearchCV on the training set after splitting the data
 The function outputs the metric results and confusion matrix'''


def fn_plot_confusion_matrix(dataset, classifier, fold):
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    # Loading a dataset
    X = dataset.data
    y = dataset.target
    class_names = dataset.target_names

    obj_classifier = get_classifier(classifier)
    parameter1 = []

    if classifier == 'c1':
        parameter1 = [{'n_neighbors': [1, 2, 3, 4, 5]}]
    elif classifier == 'c2':
        parameter1 = [{'C': [0.5, 0.1, 1, 5, 10]}]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    # load grid search cross validation function for parameter estimation
    gscv_classifier = GridSearchCV(
        estimator=obj_classifier,
        param_grid=parameter1,
        cv=fold,  # k-fold cross validation
        scoring='accuracy'
    )

    # This function returns classifier gscv_classifier and we use it to train the training set

    gscv_classifier.fit(X_train, y_train)

    # 11 Printing metric results and getting the best parameter in the output folder with text file.

    # Get parameter values, scores (accuracies) and best parameter from this gscv_classifier classifier

    object_in = []
    my_print(object_in, "---------------------------------Best_Parameters ----------------------------")

    my_print(object_in, "Grid scores on validation set:")
    means = gscv_classifier.cv_results_['mean_test_score']
    stds = gscv_classifier.cv_results_['std_test_score']
    results = gscv_classifier.cv_results_['params']
    for mean, std, param in zip(means, stds, results):
        my_print(object_in, "Parameter: %r, accuracy: %0.3f (+/-%0.03f)" % (param, mean, std * 2))

    my_print(object_in, '{0} and {1}'.format('Best parameter:', gscv_classifier.best_params_))

    # This gscv_classifier classifier now applies the best parameter, so we just use it to test the testing dataset

    y_pred = gscv_classifier.predict(X_test)

    # Plot confusion matrix and accuracy

    accuracy = metrics.accuracy_score(y_test, y_pred) * 100
    plotcm = metrics.plot_confusion_matrix(gscv_classifier, X_test, y_test, display_labels=class_names)
    plotcm.ax_.set_title('Accuracy = {0:.2f}%'.format(accuracy))
    plt.show()
    save_as(object_in, 'best_parameters')

# 12 Function to execute the run button.
'''The run button takes the command as the lambda function carrying the value of 3 variables var1,2,3 for selecting
 dataset,classifier and k-fold.
 The values are then fed to functions for plotting cv score(defined in step 9)
  and plotting confusion matrix(defined in step 10)'''


def run(var, var1, var2):
    # init dataset, classifier, parameter
    dataset = select_d(var)
    classifier = select_c(var1)
    fold = select_k(var2)

    # call some logic
    fn_cv_score(dataset, classifier, fold)
    fn_plot_confusion_matrix(dataset, classifier, fold)

    # call output
    # button_clicked(dataset, classifier, parameter)
    # xx = display_fn1(cv, "frame4")
    # yy = display_fn2(cm, "frame5")
    # print(xx)
    # print(yy)


# 13 Function to display the main window with the run button and exit button.
'''The run button takes the command as the lambda function carrying the value of 3 variables var1,2,3 for selecting
dataset,classifier and k-fold.
 The exit button takes command as window.destroy which closes the window'''


def main():
    print("Proj for Data Science...")
    # Header
    window = tk.Tk()
    dimensions = "400x700+100+100"
    myfont = frame_formatter(window, dimensions)

    # display each frames
    dataset = frame_one(window, myfont)
    classifier = frame_two(window, myfont)
    fold = frame_three(window, myfont)

    # Add Run button
    run_button = tk.Button(window, text="Run", fg="white", bg="darkblue", width=10, height=1, font=myfont,
                           command=lambda: run(dataset, classifier, fold))
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
    four_button = tk.Button(text="F4", fg="white", bg="darkblue", width=10, height=1, font=myfont)
    four_button.place(x=300, y=580)
    four_button.grid(row=3, column=1, rowspan=5)
    # dummy frame 5
    five_button = tk.Button(text="F5", fg="white", bg="darkblue", width=10, height=1, font=myfont)
    five_button.place(x=650, y=580)
    five_button.grid(row=3, column=2, rowspan=5)

    # loop again for new inputs / persistence
    window.mainloop()


# 14 Conditional statement to run the main function first as defined above.
'''It is also a way to store code that should only run when this file is executed as a script not as a module'''
if __name__ == "__main__":
    main()
