# KKL – Flow Matching

Code accompanying the paper "Generative KKL Observer for Indistinguishable Systems."

## Generate Figures

Figures from the paper can be reproduced with:
```
python main_KKL_CFM.py --epoch 0 --dataset [MODEL] --noise_std [NOISE] --name [NAME]
```
where:
- `MODEL` ∈ {`VDP`, `Test`, `Duffing`}
- `NOISE` and `NAME` can be chosen as:
  - `(0., noiseless)`
  - `(1., noisy)`

Example:
```
python main_KKL_CFM.py --epoch 0 --dataset VDP --noise_std 0. --name noiseless
```


## Training

To retrain the models, simply remove the `--epoch` argument:
```
python main_KKL_CFM.py --dataset VDP --noise_std 0. --name noiseless
```
