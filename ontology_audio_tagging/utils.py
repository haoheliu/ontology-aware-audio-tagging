import pickle

def save_pickle(obj, fname):
    print("Save pickle at " + fname)
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    print("Load pickle at " + fname)
    with open(fname, "rb") as f:
        res = pickle.load(f)
    return res