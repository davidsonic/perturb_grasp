#!/usr/bin/env python3
"""Server for multithreaded (asynchronous) chat application."""
from socket import AF_INET, socket, SOCK_STREAM
from threading import Thread


def accept_incoming_connections():
    """Sets up handling for incoming clients."""
    while True:
        client, client_address = SERVER.accept()
        print("%s:%s has connected." % client_address)
        addresses[client] = client_address
        Thread(target=handle_client, args=(client,)).start()


def handle_client(client):  # Takes client socket as argument.
    """Handles a single client connection."""
    global clients
    name = client.recv(BUFSIZ).decode("utf8")
    clients[name] = client
    broadcast('%s has connected' % name)
    while True:
        msg = client.recv(BUFSIZ)
        if msg != bytes("quit", "utf8") and msg!=bytes('', 'utf8'):
            print('force be sent:', msg)
            if 'RL' in clients:
                clients['RL'].send(msg)
        else:
            client.close()
            del clients[name]
            broadcast("%s has disconnected" % name)
            print("%s has disconnected" % name)
            break


def broadcast(msg):  # prefix is for name identification.
    """Broadcasts a message to all the clients."""
    for name in clients:
        # print('broadcast msg: ', msg)
        clients[name].send(bytes(msg, "utf8"))


clients={}
addresses = {}

HOST = ''
PORT = 33000
BUFSIZ = 1024
ADDR = (HOST, PORT)

SERVER = socket(AF_INET, SOCK_STREAM)
SERVER.bind(ADDR)

if __name__ == "__main__":
    SERVER.listen(5)
    print("Waiting for connection...")
    ACCEPT_THREAD = Thread(target=accept_incoming_connections)
    ACCEPT_THREAD.start()
    ACCEPT_THREAD.join()
    SERVER.close()