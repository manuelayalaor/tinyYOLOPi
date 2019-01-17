class Node:
    data = None
    nxt = None

    def __init__(self, data=None, nxt=None):
        self.data = data
        self.nxt = nxt

    def getElement(self):
        return self.data

    def getNext(self):
        return self.nxt



class LinkedList:
    head = None
    size = None

    def __init__(self,node=Node()):
        self.head = node
        self.size = 0
        return

    def print_list(self):
        ptr = self.head
        while ptr is not None:
            print(ptr.getElement())
            ptr = ptr.getNext()

    def empty(self):
        return self.size == 0

    def first(self):
        return self.head

    def size(self):
        return self.size

    def is_in_list(self, node):
        in_list = False
        ptr = self.head
        while ptr is not None:
            if ptr is node:
                in_list = True
                break
            ptr = ptr.getNext()
        return in_list

    def add_last(self, node):
        ptr = self.head
        while ptr is not None:
            ptr = ptr.getNext()

        ptr.nxt = node

    def remove(self, node):
        if not self.is_in_list(node):
            return None
        ptr = self.head
        nxt = self.head.getNext()
        etr = None
        while nxt is not node:
            ptr = nxt
            nxt = nxt.getNext()

        etr = nxt.getElement()
        ptr.nxt = nxt.getNext()

        nxt.data = None
        nxt.nxt = None

        return etr


