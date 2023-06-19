BASED ON `https://gitlab.com/chlohe/shielded-dreams`.

First, to install the environments used for this experiment:
```
cd safety_environments
pip install -e safety_environments
```

To start the training, run
```
python train.py --path <path_to_config>
```

To test, run
```
python test.py --path <path_to_config> --models <path_to_model>
```

Some visualisations are stored in `viz.ipynb`.