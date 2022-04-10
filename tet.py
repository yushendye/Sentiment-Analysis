import pickle

f = open("data.pickle", "rb")
p = pickle.load(f)
print(p)
