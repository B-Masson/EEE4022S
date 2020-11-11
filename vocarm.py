#VOCARM Project
#Application file to run and extensively test CNN model
#OS Version: Linux
#Richard Masson
import vocalengine as voc
import os
import time

model = voc.engineSetup("Parameters")
print("Model loaded: ", end='')
avetime = 0
avep = 0
count = 0
while True:
    user = input("Press SPACE to record command. Press t to run test. Press c to clear. Press q to quit.\n")
    if (user == ' '):
        count += 1
        predstr, pretime = voc.predict(model)
        pred = int(predstr)
        avep += pretime
        print("Prediction: Class", pred)
		# Next line requires that it point to the location of voice_command.py
        commandline = "/home/richard/ros_ws/src/arm_bringup/scripts/voice_command.py " +str(pred)
        tic = time.perf_counter()
        os.system(commandline)
        toc = time.perf_counter()
        runtime = round((toc-tic),4)
        print("Pre-processing time:", pretime)
        print("Execution time:", runtime)
        avetime += runtime
    elif (user == 't'):
        voc.test(model)
    elif (user == 'c'):
        os.system("clear")
    elif (user == 'q'):
        break
    else:
        print("Unrecognisable command. Please try again.")
if avep != 0:
    print("Average pre-processing time", round((avep/count),2))
if avetime != 0:
    print("Average time:", round((avetime/count),2))
print("Closing...")