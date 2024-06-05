from tkinter import *
import os
import threading

def d_dtcn():
    root = Tk()
    root.configure(background="white")

    def function1():
        os.system("python drowsiness_detection.py --shape_predictor shape_predictor_68_face_landmarks.dat")
        exit()

    def function2():
        os.system("python laneDetection.py")
        exit()

    def function3():
        # Create threads for each function
        thread1 = threading.Thread(target=function1)
        thread2 = threading.Thread(target=function2)

        # Start the threads
        thread1.start()
        thread2.start()

        # Wait for both threads to complete
        thread1.join()
        thread2.join()

    root.title("DROWSINESS AND LANE DETECTION")
    Label(root, text="DROWSINESS AND LANE DETECTION", font=("times new roman", 20), fg="black", bg="aqua", height=2).grid(row=2, rowspan=2, columnspan=5, sticky=N+E+W+S, padx=5, pady=10)
    Button(root, text="Run DROWSINESS DETECTION", font=("times new roman", 20), bg="#0D47A1", fg='white', command=function1).grid(row=5, columnspan=5, sticky=W+E+N+S, padx=5, pady=5)
    Button(root, text="Run LANE DETECTION", font=("times new roman", 20), bg="#0D47A1", fg='white', command=function2).grid(row=7, columnspan=5, sticky=W+E+N+S, padx=5, pady=5)
    Button(root, text="Run BOTH", font=("times new roman", 20), bg="#0D47A1", fg='white', command=function3).grid(row=9, columnspan=5, sticky=W+E+N+S, padx=5, pady=5)
    Button(root, text="Exit", font=("times new roman", 20), bg="#0D47A1", fg='white', command=root.destroy).grid(row=11, columnspan=5, sticky=W+E+N+S, padx=5, pady=5)

    root.mainloop()

# Call the function to launch the GUI
d_dtcn()
