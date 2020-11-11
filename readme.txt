README:

To make use of this entire system, you need Linux to make use of ROS Melodic (at least for the current implementation).
I've included the Windows code. It's not recommended, but you can use it to:
>Train the model
>Test basic predictions
But the actual simulation interation is only available using the required ROS environment.
Code in "Final Version (Linux)" folder is considered true final product.

To train:
Run cnn.py > Make sure you point it to the relevant directory and have that directory set up as expected.
Also, set Validate = True to enable validation functions.

To run:
Run vocarm.py > Windows version is VERY limited. Recommend you're on Linux at this point
Make sure you have a params file in /Parameters, generated via training.

For simulation:
ROS environment needs to be set up as here: https://github.com/matthewmarkey44/eee4022f
Follow his instructions exactly. Then place voice_command.py into the scripts folder in arm_bringup, and point vocarm.py there.

Notes:
vocalenginer.py > Take note of whether you're enabling trimming or not. Only needed if recording duraiton set to > 1 in recordaudioL.