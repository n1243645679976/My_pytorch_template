import kaldiio
class KaldiReader:
    def __init__(self, rspecifier, return_shape=False, segments=None):
        self.rspecifier = rspecifier
        self.return_shape = return_shape
        self.segments = segments

    def __iter__(self):
        with kaldiio.ReadHelper(self.rspecifier, segments=self.segments) as reader:
            for key, array in reader:
                if self.return_shape:
                    array = array.shape
                yield key, array


