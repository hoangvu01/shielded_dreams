BASED ON https://github.com/yusukeurakami/dreamer-pytorch and https://github.com/ajlangley/cpo-pytorch

To run the experiments, run main.py. The default is GridWorld with ABPS.

To run the other experiments, you will need to switch out the shield imports. The same goes for other 
E.g. to use a classical shield, you'll need to import the GridWorldShield class and its associated ShieldBatcher.

You can use the --symbolic-env flag to run on the symbolic environment (but don't forget to update the hyperparameters accordingly!)

CPO experiments can be found in the cpo-pytorch folder.

TODO: Tidy up
