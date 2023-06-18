import pandas as pd
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

df = pd.read_csv('wine.data',sep=',')
X = df.drop('class',axis=1)
y = df['class']
X = preprocessing.normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=2000)

LR = LogisticRegression()

LR.fit(X_train,y_train)
pred = LR.predict(X_test)
loo = LeaveOneOut()

predicitions = LR.predict(X_test)
accuracy =accuracy_score(y_test,pred)

def addNewData():
    newWindow = tk.Toplevel(root)
    newWindow.title("Dodaj nowe dane")
    classLabel = ttk.Label(newWindow,text="(nie wymaga uzupelninia przy predykcji) \nclass:")
    classLabel.pack()
    classEntry = ttk.Entry(newWindow)
    classEntry.pack()
    otherLabel1 = ttk.Label(newWindow,text="Alcohol:")
    otherLabel1.pack()
    otherEntry1 = ttk.Entry(newWindow)
    otherEntry1.pack()
    otherLabel2 = ttk.Label(newWindow,text="Malicacid:")
    otherLabel2.pack()
    otherEntry2 = ttk.Entry(newWindow)
    otherEntry2.pack()
    otherLabel3 = ttk.Label(newWindow,text="Ash:")
    otherLabel3.pack()
    otherEntry3 = ttk.Entry(newWindow)
    otherEntry3.pack()
    otherLabel4 = ttk.Label(newWindow,text="Alcalinity_of_ash:")
    otherLabel4.pack()
    otherEntry4 = ttk.Entry(newWindow)
    otherEntry4.pack()
    otherLabel5 = ttk.Label(newWindow,text="Magnesium:")
    otherLabel5.pack()
    otherEntry5 = ttk.Entry(newWindow)
    otherEntry5.pack()
    otherLabel6 = ttk.Label(newWindow,text="Total_phenols:")
    otherLabel6.pack()
    otherEntry6 = ttk.Entry(newWindow)
    otherEntry6.pack()
    otherLabel7 = ttk.Label(newWindow,text="Flavanoids:")
    otherLabel7.pack()
    otherEntry7 = ttk.Entry(newWindow)
    otherEntry7.pack()
    otherLabel8 = ttk.Label(newWindow,text="Nonflavanoid_phenols:")
    otherLabel8.pack()
    otherEntry8 = ttk.Entry(newWindow)
    otherEntry8.pack()
    otherLabel9 = ttk.Label(newWindow,text="Proanthocyanins:")
    otherLabel9.pack()
    otherEntry9 = ttk.Entry(newWindow)
    otherEntry9.pack()
    otherLabel10 = ttk.Label(newWindow,text="Color_intensity:")
    otherLabel10.pack()
    otherEntry10 = ttk.Entry(newWindow)
    otherEntry10.pack()
    otherLabel11 = ttk.Label(newWindow,text="Hue:")
    otherLabel11.pack()
    otherEntry11 = ttk.Entry(newWindow)
    otherEntry11.pack()
    otherLabel12 = ttk.Label(newWindow,text="0D280_0D315_of_diluted_wines:")
    otherLabel12.pack()
    otherEntry12 = ttk.Entry(newWindow)
    otherEntry12.pack()
    otherLabel13 = ttk.Label(newWindow,text="Proline:")
    otherLabel13.pack()
    otherEntry13 = ttk.Entry(newWindow)
    otherEntry13.pack()
    def add():
        global df
        global accuracy
        data = []
        newClass = int(classEntry.get())
        data.append(float(otherEntry1.get()))
        data.append(float(otherEntry2.get()))
        data.append(float(otherEntry3.get()))
        data.append(float(otherEntry4.get()))
        data.append(float(otherEntry5.get()))
        data.append(float(otherEntry6.get()))
        data.append(float(otherEntry7.get()))
        data.append(float(otherEntry8.get()))
        data.append(float(otherEntry9.get()))
        data.append(float(otherEntry10.get()))
        data.append(float(otherEntry11.get()))
        data.append(float(otherEntry12.get()))
        data.append(float(otherEntry13.get()))

        row = pd.DataFrame([[newClass]+data],columns=df.columns)
        df = pd.concat([df,row],ignore_index=True)
        def refresh():
            treeview.delete(*treeview.get_children())
            for index,row in df.iterrows():
                values = tuple(row)
                treeview.insert("","end",values=values)
        X = df.drop('class',axis=1)
        y = df['class']
        X = preprocessing.normalize(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2000)
        LR =LogisticRegression()

        LR.fit(X_train,y_train)
        prediciton = LR.predict(X_test)
        accuracy = accuracy_score(y_test,prediciton)
        accuracyLabel.configure(text=f"Dokładność modelu: {accuracy}")

        refresh()
        messagebox.showinfo("Informacja",
                            f"Nowe dane zostały dodane.\nDokładność modelu po ponownym budowaniu: {accuracy}")
    def predict():
        data = []
        data.append(float(otherEntry1.get()))
        data.append(float(otherEntry2.get()))
        data.append(float(otherEntry3.get()))
        data.append(float(otherEntry4.get()))
        data.append(float(otherEntry5.get()))
        data.append(float(otherEntry6.get()))
        data.append(float(otherEntry7.get()))
        data.append(float(otherEntry8.get()))
        data.append(float(otherEntry9.get()))
        data.append(float(otherEntry10.get()))
        data.append(float(otherEntry11.get()))
        data.append(float(otherEntry12.get()))
        data.append(float(otherEntry13.get()))
        prediciton = LR.predict([data])
        messagebox.showinfo("Predykacja",f"Powinnien nalezec do klasy: {prediciton[0]}")
    commitButton  = tk.Button(newWindow,text="Dodaj i przebuduj model",command=add)
    commitButton.pack()
    predButton = tk.Button(newWindow,text="Predykuj",command=predict)
    predButton.pack()
def graph():
    plt.figure(figsize=(12,10))
    columns = df.columns[1:]
    num_graph = len(columns)
    row = int(np.sqrt(num_graph))
    column = int(np.ceil(num_graph/row))
    for i, col in enumerate(columns):
        plt.subplot(row,column,i+1)
        classes = df['class'].unique()
        color = ['orange','blue','purple']
        for j, Class in enumerate(classes):
            Class_data = df[df['class']==Class]
            plt.scatter(Class_data[col],Class_data[col],color=color[j],label=f'Klasa {Class}')
        plt.xlabel(col)
        plt.ylabel(col)
        plt.title(f'{col} data')
        plt.legend()
    plt.tight_layout()
    plt.show()


root = tk.Tk()
root.title("Logistyczna regresja")
root.geometry("1000x800")

treeview = ttk.Treeview(root)
treeview["height"] = 30
treeview["columns"] = tuple(df.columns)
for column in df.columns:
    treeview.heading(column,text=column)
    treeview.column(column,width=80)
for index, row in df.iterrows():
    values = tuple(row)
    treeview.insert("","end",values=values)
treeview.pack()
addData = tk.Button(root, text="Dodaj dane",command=addNewData)
addData.pack()
showGraph = tk.Button(root,text="Wykres",command=graph)
showGraph.pack()
accuracyLabel = ttk.Label(root, text=f"Dokładność modelu: {accuracy}")
accuracyLabel.pack()

root.mainloop()