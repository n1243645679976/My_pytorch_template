# README
## Feature Extraction
### Config
```
dataset:
    feature: ['spectrogram#wav.scp', 'spectrogram1#wav1.scp', 'stoi#trial#wav.scp#trial.scp', 'pesq#trial#wav.scp#trial.scp']
    exclude_inference: ['score.txt']
    collate_fn: 'pad_to_max'
    batch_size: 1
```
This will extract spectrogram by `conf/spectrogram.yaml` from `data/{data}/wav.scp}`, by `conf/spectrogram1.yaml` from `data/{data}/wav1.scp`, by `conf/stoi.yaml` from `data/{data}/trial` with `data/{data}/wav.scp` and `data/{data}/trial.scp`, by `conf/pesq.yaml` from `data/{data}/trial` with `data/{data}/wav.scp` and `data/{data}/trial.scp`.
## Dataset
