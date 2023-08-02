# CCR
Implementaion of the paper "Counterfactual Cross-modality Reasoning for Weakly Supervised Video Moment Localization" (ACM MM 2023).

## Data preparation
Features for [Charades-STA](https://github.com/JonghwanMun/LGI4temporalgrounding) (by [LGI](https://github.com/JonghwanMun/LGI4temporalgrounding)) and for [ActivityNet Captions](http://activity-net.org/challenges/2016/download.html). Download these features and put them into the corresponding directories in "\data". Refering to [CPL](https://github.com/minghangz/cpl) for more details.

## Training and inference

### Train
```bash
# [dataset]: charades, activitynet
python train.py --config-path config/[dataset]/main.json --log_dir LOG_DIR --tag TAG
```
### Inference
```bash
python train.py --config-path CONFIG_FILE --resume CHECKPOINT_FILE --eval
```
