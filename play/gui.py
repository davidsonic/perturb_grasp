import tkinter as tk
from tkinter import *

from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread
import time
import numpy as np

# window application########################################################################
window = tk.Tk()
window.title('my window')
window.geometry('500x500')

##########################################################################

def print_selection1(v):
    f1.config(text=v)

def print_selection2(v):
    f2.config(text=v)

def print_selection3(v):
    f3.config(text=v)

# label for force
fr0=Frame(window)
fr0.pack(fill=X)
l0=tk.Label(fr0, bg="yellow", text="Force")
l0.pack(fill=X)

fr1=Frame(fr0)
fr1.pack(fill=X)

f1 = tk.Label(fr1, bg="white", width=20, text="empty")
f1.pack(padx=5, pady=10, side=LEFT)

f2 = tk.Label(fr1, bg="white", width=20, text="empty")
f2.pack(padx=5, pady=10, side=LEFT)

f3 = tk.Label(fr1, bg="white", width=20, text="empty")
f3.pack(padx=5, pady=10, side=LEFT)

fr2=Frame(fr0)
fr2.pack(fill=X)

# scale widget
s1 = tk.Scale(fr2, label='', from_=0, to=1, orient=tk.VERTICAL, width=20,
             length=100, showvalue=1, tickinterval=0.01, resolution=0.01, command=print_selection1)
s1.pack(padx=5, pady=10, side=LEFT)


s2 = tk.Scale(fr2, label='', from_=0, to=1, orient=tk.VERTICAL, width=20,
             length=100, showvalue=1, tickinterval=0.01, resolution=0.01, command=print_selection2)
s2.pack(padx=55, pady=10, side=LEFT)


s3 = tk.Scale(fr2, label='', from_=0, to=1, orient=tk.VERTICAL, width=20,
             length=100, showvalue=1, tickinterval=0.01, resolution=0.01, command=print_selection3)
s3.pack(padx=50, pady=10, side=LEFT)


#######################################################################

def print_selection4(v):
    f4.config(text=v)

def print_selection5(v):
    f5.config(text=v)

def print_selection6(v):
    f6.config(text=v)

# label for torque
fr2=Frame(window)
fr2.pack(fill=X)
l1=tk.Label(fr2, bg="yellow", text="Torque")
l1.pack(fill=X)

fr3=Frame(fr2)
fr3.pack(fill=X)


f4 = tk.Label(fr3, bg="white", width=20, text="empty")
f4.pack(padx=5, pady=10, side=LEFT)

f5 = tk.Label(fr3, bg="white", width=20, text="empty")
f5.pack(padx=5, pady=10, side=LEFT)

f6 = tk.Label(fr3, bg="white", width=20, text="empty")
f6.pack(padx=5, pady=10, side=LEFT)

fr4=Frame(fr2)
fr4.pack(fill=X)

# scale widget
s4 = tk.Scale(fr4, label='', from_=0, to=1, orient=tk.VERTICAL, width=20,
             length=100, showvalue=1, tickinterval=0.01, resolution=0.01, command=print_selection4)
s4.pack(padx=5, pady=10, side=LEFT)


s5 = tk.Scale(fr4, label='', from_=0, to=1, orient=tk.VERTICAL, width=20,
             length=100, showvalue=1, tickinterval=0.01, resolution=0.01, command=print_selection5)
s5.pack(padx=55, pady=10, side=LEFT)


s6 = tk.Scale(fr4, label='', from_=0, to=1, orient=tk.VERTICAL, width=20,
             length=100, showvalue=1, tickinterval=0.01, resolution=0.01, command=print_selection6)
s6.pack(padx=50, pady=10, side=LEFT)


#####################################################################
def btn_confirm_press(event):
    l0.config(bg='red')
    l1.config(bg="red")
    # change parameters in the program


def btn_confirm_release(event):
    l0.config(bg='yellow')
    l1.config(bg="yellow")
    send_msg()


def btn_clear():
    f1.config(text="")
    f2.config(text="")
    f3.config(text="")
    f4.config(text="")
    f5.config(text="")
    f6.config(text="")


def destroy():
    global window
    global lb
    lb.config(text='quit')
    send_msg()
    time.sleep(0.5)
    window.destroy()



def btn_gen_random():
    # gen random force within range, TODO
    rnd = np.random.uniform(-3,3,6)
    f1.config(text = rnd[0])
    f2.config(text = rnd[1])
    f3.config(text = rnd[2])
    f4.config(text = rnd[3])
    f5.config(text = rnd[4])
    f6.config(text = rnd[5])



fr4=Frame(window)
fr4.pack(fill=X)

lb=tk.Label(fr4, bg="white", width=20, text="")
lb.pack(fill=X)

b1=tk.Button(fr4, text="Confirm")
b1.bind("<ButtonPress>", btn_confirm_press)
b1.bind("<ButtonRelease>", btn_confirm_release)
b1.pack(side=LEFT)

b4=tk.Button(fr4, text="Random", command=btn_gen_random).pack(side=LEFT)
b2=tk.Button(fr4, text="Clear", command=btn_clear)
b2.pack(side=LEFT)

b3=tk.Button(fr4, text="Quit", command=destroy).pack(side=LEFT)



###################################
# networking
def receive():
    global client_socket
    global window
    while True:
        try:
            msg = client_socket.recv(BUFSIZ).decode('utf8')
            if msg!='':
                print('receive msg: ', msg)
            else:
                window.quit()
        except OSError:
            break


def send_msg(evt=None):
    global client_socket
    global lb
    lbtext=lb.cget('text')
    if lbtext == 'quit':
        print(lbtext)
        client_socket.send(bytes('quit','utf8'))
        return

    forces=[]
    forces.append(f1.cget("text"))
    forces.append(f2.cget("text"))
    forces.append(f3.cget("text"))
    forces.append(f4.cget("text"))
    forces.append(f5.cget("text"))
    forces.append(f6.cget("text"))
    if isinstance(forces[0], float):
        forces = list(map(str, forces))
    forces=','.join(forces)

    client_socket.send(bytes(forces, 'utf8'))



HOST = 'localhost'
# PORT=input('Input Port: ')
PORT = '33000'
PORT = int(PORT)

BUFSIZ=1024
ADDR=(HOST, PORT)

client_socket = socket(AF_INET, SOCK_STREAM)
client_socket.connect(ADDR)
client_socket.send(bytes('gui','utf8'))
receive_thread = Thread(target=receive)
receive_thread.start()

window.mainloop()