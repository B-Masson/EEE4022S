README:

To make use of this entire system, you need Linux to make use of ROS Melodic (at least for the current implementation).

To train:
Run cnn.py > Make sure you point it to the relevant directory and have that directory set up as expected.
Also, set (Validate = True) to enable validation functions.

To run:
Run vocarm.py > Requires Linux and ROS simulation setup.
There is a line you can comment out to detach the sim aspect from the prediction model.
Make sure you have a params file in /Parameters, generated via training.

For simulation:
ROS environment needs to be set up as here: https://github.com/matthewmarkey44/eee4022f
Follow his instructions exactly. Then place voice_command.py into the scripts folder in arm_bringup, and point vocarm.py there.

Notes:
vocalengine.py > Take note of whether you're enabling trimming or not. Only needed if recording duraiton set to > 1 in recordaudioL.
I have no uploaded pre-trained parameters since those are actually much bigger files than you would think.
