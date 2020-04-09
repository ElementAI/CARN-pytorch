## DATASETS
* Please download it from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB). Unpack the tar file to any place you want. Then, change the dir_data argument in src/option.py to the place where DIV2K images are located

* Urban100, Manga, Set5, Set15: https://cv.snu.ac.kr/research/EDSR/benchmark.tar

## Running the code
`python3 trainval.py -e AWSDR_SDx2_edgar -sb ./directory/to/save/logs -j 0 -r 1 # -j is for cluster, -r is for reset or start from last checkpoint`

The experiment you want to run is defined with `-e`

## Defining an experiment
Modify:
`exp_configs/edgar.py`

## Dependencies
`pip install --upgrade git+https://github.com/ElementAI/haven`

## TODO Edgar
* Modify `models/kornia_trainer.py`, add the corresponding losses (commented in the code)
