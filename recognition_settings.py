# setup recognition
LOUD_THRESHOLD = 25000  # peak level used to preliminary detect melody
SILENCE_THRESHOLD = 3000  # all frames with lower volume than this threshold will be handled as silence
ROUGH_SEARCH_INTERVAL = 5  # chunks of audio for initial rough analysis (in seconds)
RATE = 22050  # recommended recording bit rate
MIN_SILENCE_ON_BORDERS = 0.7  # minimum silence before and after melodies  (in seconds)
BEG_END_ACCURACY = 10  # accuracy when searching for melody beginnings and ends (in frames). Affects performance
SILENCE_BETWEEN_REPEATS = 0.4  # distance between
BELOW_LOUDEST_PEAK = 0.7  # when counting repeats, we take the melody's loudest frame, reduce it and use as search criteria for peaks
MAX_REPEATS = 5  # max number of repeats of a melody within one alert
MELODY_LENGTH_DEVIATION = 0.4  # max acceptable difference in length between a sample and recognized melody

# recognition shortcuts
SILENCE_BORDERS = int(RATE * MIN_SILENCE_ON_BORDERS)  # minimum silence before and after melodies (in frames)
ROUGH_SEARCH = int(RATE * ROUGH_SEARCH_INTERVAL)  # chunks of audio for initial rough analysis (in frames)


# file setup
RECORD_PATH = 'Records'  # the big record will be stored here
MELODY_PATH = 'Melodies'  # the detected melodies will be stored here
SAMPLES_PATH = 'Samples'  # samples for melody identification should be stored here
RECORD_FILE_NAME = 'record_{}.wav'
MELODY_FILE_NAME = 'melody_{}.wav'

# file shortcuts
RECORD = '\\'.join([RECORD_PATH, RECORD_FILE_NAME])
MELODY = '\\'.join([MELODY_PATH, MELODY_FILE_NAME])
