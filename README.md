# AI Controller

This repo contains the ai controller for our floatsat project.

The file `trainer.py` should be run locally to train the model.

The file `tester.py` can be used to test the model.

The file `AI_Controller.py` can be run on the pi together with the stm  board to run the ai controller.

The file `FloatSatEnv.py` contains the environment that is used to train the model.

## Setup

### Development
The following steps describe how to set up the development environment on linux:
1. Clone the Basilisk repo as described here: `https://hanspeterschaub.info/basilisk/Install/pullCloneBSK.html`

1. `cd basilisk`
1. Clone this repo into the basilisk folder: `git clone https://github.com/NickiBroichhausen/FloatSat_AI_Controller.git`
1. `cd FloatSat_AI_Controller`
1. Make sure you activated your venv as described in the second step and then run `pip install -r requirements.txt`
1. Test the installation by running `python tester.py` 

### Satellite
The following steps describe how to set up the environment on the pi:
1. Clone the following repo onto the satellite to handle the communication: `git clone --recurse-submodules https://github.com/NickiBroichhausen/starmapper.git` (This will also clone the AI module as submodule)
1. Set up the RODOS communication with the stm board. For that clone RODOS onto the pi in a different directory: `git clone https://gitlab.com/rodos/rodos`
1. Install the python middleware with pip from the RODOS folder `support/support-programs/middleware-python/`
1. Configure the auto start: `sudo nano /etc/systemd/system/my_script.service`
```
[Unit]
Description=My Python Script with Xvfb
After=network.target

[Service]
Type=simple
WorkingDirectory=</path/to/your/script>
ExecStartPre=/bin/bash -c '/usr/bin/Xvfb :1 -screen 0 800x600x16 & export DISPLAY=:1'
ExecStart=/usr/bin/python3 </path/to/your/script/your_script.py>
Restart=always
User=<your_username>
Environment=DISPLAY=:1

[Install]
WantedBy=multi-user.target

```
Replace the values in <> with the values of the `main_rxtx.py` file in the main repo.

## WARNING
For this to run on a pi, "numpy<=1.26.4" has to be installed. Otherwise the models cannot be loaded.
