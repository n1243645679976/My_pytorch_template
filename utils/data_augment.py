import os
import random
import torchaudio as ta
class DataAugment():
    def __init__(self, args, sampling_rate, data, preprocess_list, preprocess_conf):
        self.sampling_rate = sampling_rate
        self.data = data
        self.preprocess_list = preprocess_list
        self.preprocess_conf = preprocess_conf
        self.preprocess = []
        for augment in self.preprocess_list:
            if augment == 'speech_augment':
                self._speech_augment_max = self.preprocess_conf['speech_augment']['max']
                self._speech_augment_min = self.preprocess_conf['speech_augment']['min']
            if augment == 'trim_add_silence':
                self._trim_add_silence_max = self.preprocess_conf['trim_add_silence']['max']
                self._trim_add_silence_min = self.preprocess_conf['trim_add_silence']['min']
        
    def forward(self, key, x):
        for augment in self.preprocess_list:
            if augment == 'speech_augment':
                x = self.speech_augment(x)
            elif augment == 'trim_add_silence':
                x = self.trim_add_silence(key, x)
            else:
                raise NotImplementedError(f'Data augment {augment} not Implemented')

    def speed_augment(self, x):
        speed_coef = random.random() * (self._speech_augment_max - self._speech_augment_min) + self._speech_augment_min
        effects = [['speed', str(speed_coef)], ['rate', str(self.sampling_rate)]]
        x = ta.sox_effects.apply_effects_tensor(x, self.sampling_rate, effects)[0]
        return x

    def trim_add_silence(self, key, x):
        if not hasattr(self, 'silences'):
            with open(os.path.join('data', self.data, 'silence')) as f:
                for line in f.read().splitlines():
                    if line.split() > 1:
                        _key, value = line.split(' ', 1)
                        values = list(map(float, value.split()))
                        self.silences[_key] = \
                            list([(start_pos, silence_len) for start_pos, silence_len in zip(values[::2], values[1::2])])
                    else:
                        self.silences[line] = None

        if self.silences[key] != None:
            start_pos, silence_len = random.choice(self.silences[key])
            silence_insert = random.random() * silence_len + start_pos
            silence_insert_len = random.random() * (self._trim_add_silence_max - self._trim_add_silence_min) + self._trim_add_silence_min
            effects = [['pad', '{silence_insert_len}@{silence_insert}']]
        else:
            insert_start = random.choice([0, 1])
            silence_insert_len = random.random() * (self._trim_add_silence_max - self._trim_add_silence_min) + self._trim_add_silence_min
            effects = [['pad', '0', '0']]
            effects[0][insert_start+1] = str(silence_insert_len)
        x = ta.sox_effects.apply_effects_tensor(x, self.sampling_rate, effects)[0]
        return x
