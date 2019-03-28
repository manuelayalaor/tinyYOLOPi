import jsonify

class Serializer():

    def __init__(self, model):
        '''Receives model'''
        self.model = model
        return
    #Context Manager Logic
    def __enter__():
        pass
    def __exit__():
        pass
    ###
    
    def extract(self):
        '''Extract info from model'''
        #This is problematic
        layer = self.model.get_layer(name=None, index=None)
        #assumes we already now the depth of the Network

        config = layer.get_config()#this returns a dict
        return

    def dump(self,filename=""):
        #do json stuff here
        '''Dump the extracted info into a file as a json'''
        return
