from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename

class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()



    def create_widgets(self):
        self.filediag = Button(self)
        self.filediag["text"] = "find folder\n(click me)"
        self.filediag["command"] = self.callback
        self.filediag.pack(side="top")


        self.quit = Button(self, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.pack(side="bottom")

    def callback(self):
        name = askopenfilename()
        print(name)

root = Tk()
app = Application(master=root)
app.master.title("Beta")
app.master.maxsize(1000, 400)
app.mainloop()