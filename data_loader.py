class DataLoader:

    def __init__(self):
        self.train_X = range(100)
        self.train_Y = range(100)

    def getBatch(self, batch_size, tag):
        return self.train_X[:batch_size], self.train_Y[:batch_size]

    def _preprocess(self):
        pass

data_loader = DataLoader()
X, Y = data_loader.getBatch(16, "train")
print "train", X, Y
X, Y = data_loader.getBatch(16, "test")
print "test", X, Y
X, Y = data_loader.getBatch(16, "val")
print "val", X, Y
