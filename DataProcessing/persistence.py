import csv
import os

def retrieve_file(filename=""):

    if os.path.isfile(os.path.abspath(filename)):
        return os.path.abspath(filename)
    else:
        raise IOError(f" File {filename}: was not found")

#TODO make this a generator instead of loading all data directly
#This causes memory to fill up rapidly
def retrieve_data(name):
    '''Returns a list of dicts representing the csv filename provided'''
    #Assumes csv extension is already provided
    data=[]
    if(os.path.exists(name)):
        try:
            with open(name,'rt') as fin:
                cin = csv.DictReader(fin)
                #data = [row for row in cin]
                for row in cin:
                    yield row
        except IOError as detail:
            print('Run-time error while reading data:', detail)

    else:
        raise IOError(f'File with name {name} does not exist!')

    return data

def write_data(data={}, filename=""):
    try:
        with open(name,'wt') as fout:
            cout = csv.writer(fout,quoting=csv.QUOTE_NONE)#Likely candidate since it usually processes a list of lists
            #Confirmed must take a list of list, Tups or Dicts
            #Works fine now
            cout.writerows(data)
    except IOError as e:
        print("[#]Error while writing CSV data:", e)
        return False
    return True#success
