# AI Controller

This repo contains the ai controller for our floatsat project.

The file `trainer.py` should be run locally to train the model.

The file `tester.py` can be used to test the model.

The file `AI_Controller.py` can be run on the pi together with the stm  board to run the ai controller.

The file `FloatSatEnv.py` contains the environment that is used to train the model.

## WARNING
For this to run on a pi, "numpy<=1.26.4" has to be installed. Otherwise the models cannot be loaded.