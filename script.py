from datetime import datetime as dt
from time import sleep

from tools import *
# from gm_app_utilities import compare_vars


def melody(name, minute, repeats, volume):
    return '{}_{}_{}_{}'.format(minute, name, repeats, volume)


expected_melodies_1 = {melody(name='L',  minute=12, repeats=1, volume=36),
                       melody(name='L',  minute=9,  repeats=4, volume=100),
                       melody(name='H',  minute=3,  repeats=2, volume=62),
                       melody(name='UL', minute=6,  repeats=5, volume=83),
                       }

# start timestamp
start = dt.now()
print('Started at: {}\n'.format(start))

# record audio
rec_time = 0.05/3
current_media_volume = 36  # get the real current media volume here after debug
rec_1 = record(rec_time)

# do something in the script while audio is being recorded in the background
sleep(rec_time)

# # print the expected_melodies
# print('\nThe expected melodie are')
# for key, value in expected_melodies.items():
#     print('{} at minutes: {}'.format(key, value))

# rec_1.play()

# analise the recording and compare the results to the expected_melodies
result_1 = rec_1.assure(expected_melodies_1, debug=True)  # TODO uncomment this line
# this has to print the detected melodies names and times when they were triggered
# in the same format as expected_melodies
# and return True or False

# *optional - save the detected melodies to folder Melodies
# rec_1._save_detected_melodies()  # TODO uncomment this line

# # ensure test results
# compare_vars(result_1, True)

# end timestamp
end = dt.now()
print('\nTime elapsed: {}'.format(end-start))
