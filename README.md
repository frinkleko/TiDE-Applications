# TiDE-Applications
A unofficial pytorch implementation of "Long-term Forecasting with TiDE: Time-series Dense Encoder" and its sample code of applications

![TiDE Model](https://telegraph-image-2rp.pages.dev/file/f88b4e8f7701eee2852c3.png)

## Usage
1. Config model
   
   edit the `config.yaml` in config folder

2. Train
   
    ```bash
    python3 train.py --name name --dataset traffic 
    ```
    
    In order to use a different dataset you need to customize [data_utils.py](https://github.com/frinkleko/TiDE-Applications/blob/main/utils/data_utils.py)

## TODO
Warning: This repository is not finished now, only the network code is done. Sample code of own dataset is not finished now.

- [x] Network code
- [ ] Utils code
    - [x] basic utils
    - [x] scheduler
    - [x] data_utils
    - [ ] time series dataset
- [ ] train and eval
    - [x] logger and summary
- [ ] sample code of own dataset