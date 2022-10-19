class AdditionalDataBase():
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg

    def __call__(self, data):
        return self.process_data(data)

    def process_data(self, data):
        raise NotImplementedError

    def collate_fn(self, batch):
        return dict()

class SliceWav(AdditionalDataBase):
    def __init__(self, max_wav_seconds,cfg=None,phase=None) -> None:
        super().__init__()
        self.max_wav_len = int(max_wav_seconds*16000)
    def __call__(self, data):
        # input: [Time]
        # output: [Time]
        return data[:self.max_wav_len]
