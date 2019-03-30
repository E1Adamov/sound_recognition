from scipy.io.wavfile import read, write
from datetime import timedelta, datetime as dt
import sounddevice as sd
from matplotlib import pyplot as plt
from scipy.fftpack import fft
import pylab
import os


import recognition_settings as rs


class Melody:
    def __init__(self, np_audio):
        self.np_audio = np_audio
        self.file_name = None
        self.number = None
        self.name = None
        self.trigger_time = None
        self.volume = None
        self.repeats = None


class RecordingObject:

    def __init__(self, duration, media_volume, debug=False):
        """
        :param duration: duration of desired audio recording in seconds
        :param media_volume: current media volume at the device
        """
        self.__debug_mode = debug
        self.__media_volume = media_volume
        self.__file = rs.RECORD_FILE_NAME.format(dt.now().strftime('%Y%m%d%H%M%S'))
        self.__audio = self.__record(duration)  # at initialization, the record is stored here, later it's saved to file
        self.__duration = self.__get_duration(self.__audio)
        self.__rec_numpy_array = None  # we need to save the file and then read it to numpy array for further analysis
        self.__exp_mel_qty = None  # quantity of expected melodies quantity, used for assertions
        self.__peaks = None  # number of frame with initially found peaks
        self.__actual_mel_quant = None  # actual quantity of detected melodies. Used for assertions
        self.__beginnings = None  # number of frame when each melody was triggered
        self.__ends = None  # number of frame when each melody ends
        self.__melodies = []  # list with instances of detected melodies
        self.__melody_trigger_times = None  # time when each melody was triggered in datetime format
        self.__repeats = None  # how many times each melody was played
        self.__melodies_without_repeats = []  # detected melodies without repeats
        self.__repeat_quantities = None  # repeats of each detected melody
        self.__melody_volumes = None
        self.__recognized_melody_names = None  # identified names of triggered melodies
        a = Melody()

    def debugger(function):

        def func(*args, **kwargs):

            if args[0].__debug_mode:

                result = function(*args, **kwargs)

                return result

        return func

    @debugger
    def __assert(self, assertion, error_message=None):
        """
        Assertion that includes custom message and logging
        :param assertion: logical assertion
        :param error_message: text to be printed in case assertion fails
        :return:
        """
        try:
            assert assertion

        except AssertionError:
            raise AssertionError('\n' + error_message)

    @staticmethod
    def __plot_frequency(audio, rate=rs.RATE):
        samples = audio.shape[0]
        data_fft = fft(audio)

        fft_abs = abs(data_fft)
        freq = pylab.fftfreq(samples, 1 / rate)

        plt.xlim([10, rate / 2])
        plt.xscale('log')
        plt.grid(True)
        plt.xlabel('Frequency (Hz)')
        plt.plot(freq, fft_abs)
        plt.show()

    @staticmethod
    def __get_frequency(np_audio):
        return pylab.fft(np_audio)

    def __align_melodies(self, np_audio_1, np_audio_2):

        first_peak_index_audio_1 = self.__get_peak(np_audio_1)
        first_peak_index_audio_2 = self.__get_peak(np_audio_2)

        aligned_1 = np_audio_1[first_peak_index_audio_1:]
        aligned_2 = np_audio_2[first_peak_index_audio_2:]

        return aligned_1, aligned_2

    def __get_melody_similarity(self, np_audio_1, np_audio_2):

        loud_threshold_1 = max(np_audio_1) * rs.BELOW_LOUDEST_PEAK
        loud_threshold_2 = max(np_audio_2) * rs.BELOW_LOUDEST_PEAK

        silence = rs.SILENCE_BETWEEN_REPEATS * rs.RATE

        print('__get_melody_similarity', len(np_audio_1), len(np_audio_2))

        peaks_in_np_audio_1 = self.__get_peaks(np_audio_1, loud_threshold=loud_threshold_1, step=silence)
        peaks_in_np_audio_2 = self.__get_peaks(np_audio_2, loud_threshold=loud_threshold_2, step=silence)

        # if quantity of peaks is different -> these are different melodies
        if len(peaks_in_np_audio_1) != len(peaks_in_np_audio_2):
            return 0

        aligned_1, aligned_2 = self.__align_melodies(np_audio_1, np_audio_2)

        # if the melodies' lengths differ a lot -> these are different melodies
        deviation = rs.MELODY_LENGTH_DEVIATION * rs.RATE
        print(deviation, abs(len(aligned_1) - len(aligned_2)), )
        print(len(aligned_1), len(aligned_2), '\n')
        if abs(len(aligned_1) - len(aligned_2)) > deviation:
            return 0

        end = self.__get_min_length_from_audio_couple(aligned_1, aligned_2) - 1

        trimmed_audio_1, trimmed_audio_2 = np_audio_1[:end], np_audio_2[:end]

        frequencies_1 = self.__get_frequency(trimmed_audio_1)
        frequencies_2 = self.__get_frequency(trimmed_audio_2)

        result = pylab.corrcoef(frequencies_1, frequencies_2)
        similarity = abs(float(result[1][0]))

        return round(similarity, 3)

    @staticmethod
    def __get_min_length_from_audio_couple(np_audio_1, np_audio_2):
        min_len = min([len(np_audio_1), len(np_audio_2)])
        return min_len

    def __save_recording(self, path=rs.RECORD_PATH, rate=rs.RATE):
        """
        save numpy audio to .wav file
        :param path:
        :param rate:
        :return:
        """
        self.__clear_dir(path)

        file_ = '\\'.join([path, self.__file])

        write(file_, rate, self.__audio)

        msg = "Recording {} was not saved".format(self.__file)
        self.__assert(self.__file in self.__get_files_from_dir(path), msg)

        self.__audio = None

    def play(self, rate=rs.RATE, block=True):
        """
        will not be available after debug because self.__audio will be purged for memory optimization
        :param rate:
        :param block:
        :return:
        """
        sd.play(self.__audio, samplerate=rate, blocking=block)
        return

    @staticmethod
    def __get_duration(np_audio, rate=rs.RATE):
        """
        gets the record duration from a numpy array containing audio
        :param np_audio:
        :param rate:
        :return:
        """
        rec_seconds = len(np_audio) / rate
        rec_duration = timedelta(seconds=rec_seconds)
        return rec_duration

    @staticmethod
    def __record(duration, rate=rs.RATE, block=False):
        """
        record audio
        :param duration: in minutes
        :param rate:
        :param block: record runs in a separate thread. If block == True =>
        program will continue running only after the recording is finished
        :return: numpy array containing audio
        """
        # convert minutes to seconds
        duration = int(duration * 60)

        # convert duration to frames
        frames = int(rate * duration)

        # start recording
        record = sd.rec(frames, rate, channels=1, blocking=block)

        # TODO implement auto saving here

        return record

    @staticmethod
    def __read_from_wav_to_numpy(file_):
        """
        reads .wav file and returns its bit rate and numpy array
        """
        _, numpy_array = read(file_)

        return numpy_array

    def __get_melody_trigger_times(self, rate=rs.RATE):
        """
        :param rate:
        :return: list with melodies start time in datetime format
        """

        beginning_seconds = [round(i / rate, 2) for i in self.__beginnings]
        melody_trigger_times = [timedelta(seconds=i) for i in beginning_seconds]

        return melody_trigger_times

    def __get_repeat_quantity(self, numpy_array, melody_number, step=rs.SILENCE_BETWEEN_REPEATS):

        step = int(step * rs.RATE)

        loudest_peak = max(numpy_array)

        # TODO test this on records via a cable with fewer noise
        search_threshold = int(loudest_peak * rs.BELOW_LOUDEST_PEAK)

        peaks = []

        step_count = 0
        for i in range(0, len(numpy_array), step):

            peak = self.__get_peak(numpy_array[i:], loud_threshold=search_threshold)

            if peak is None:
                peak = 0

            peak_index = peak + int(step_count * step)
            if peak and peak_index not in peaks:
                peaks.append(peak_index)

            step_count += 1

        repeats = len(peaks)

        message = "{} repeats detected. Melody={}, length={}, step={}, peaks={}".format(repeats, melody_number, len(numpy_array), step, peaks)
        self.__assert(0 < repeats <= rs.MAX_REPEATS, message)

        return repeats

    @staticmethod
    def __get_peak(numpy_array, loud_threshold=rs.LOUD_THRESHOLD):

        for i in range(len(numpy_array)):

            if abs(numpy_array[i]) > loud_threshold:

                return i

    def __get_peaks(self, numpy_array,
                    loud_threshold=rs.LOUD_THRESHOLD,
                    step=rs.ROUGH_SEARCH,
                    accuracy=rs.BEG_END_ACCURACY):
        """

        :param numpy_array:
        :param loud_threshold:
        :return: list of peaks in format [frame_index, frame_index...]
        """

        peaks = []

        # rough search with step in several seconds (rs.ROUGH_SEARCH)
        current_search_index = 0
        for i in range(len(numpy_array)):

            peak_found = False

            if current_search_index >= len(numpy_array):
                break

            end = int(current_search_index + step)
            if end > len(numpy_array):
                end = len(numpy_array)

            # if a peak is detected within the search interval
            if max(numpy_array[current_search_index:end]) > loud_threshold:

                # search this interval thoroughly for the first peak frame
                for j in range(int(current_search_index), end, accuracy):

                    if abs(numpy_array[j]) > loud_threshold:
                        peaks.append(j)
                        current_search_index = j + step
                        peak_found = True
                        break

            if not peak_found:
                current_search_index += step  # shift search index for process optimization

        self.__actual_mel_quant = len(peaks)

        return peaks

    @staticmethod
    def __get_beginning(numpy_array, global_peak_index, global_start_index):
        """
        :param global_peak_index: the index of this small numpy_array's peak in the whole recording
        :param global_start_index: the index of this small numpy_array's beginning in the whole recording
        :param numpy_array:
        :return: frame index of the melody beginning
        """

        local_peak_index = rs.ROUGH_SEARCH  # this is the middle of the numpy_array

        # in case the peak is too close to the record beginning
        if len(numpy_array) < rs.ROUGH_SEARCH * 2:
            local_peak_index = global_peak_index

        count = 1
        # index in the current part of the numpy array
        for local_beg_index in range(local_peak_index-1, 0, -rs.BEG_END_ACCURACY):

            # search from the peak backwards to find a silent frame
            if abs(numpy_array[local_beg_index]) < rs.SILENCE_THRESHOLD:

                # if silence is long enough:
                current_piece = numpy_array[local_beg_index-rs.SILENCE_BORDERS:local_beg_index]
                if len(current_piece) == 0:
                    print("Couldn't find a beginning before peak. Increase SILENCE_THRESHOLD")
                    exit(126)

                if max(current_piece) < rs.SILENCE_THRESHOLD:

                    # index in the whole recording
                    global_beg_index = global_start_index + local_beg_index

                    return global_beg_index

    @staticmethod
    def __get_end(numpy_array, global_peak_index, silent_threshold=rs.SILENCE_THRESHOLD, silence=rs.RATE):

        # rough search with step in one second (rs.RATE)
        for i in range(0, len(numpy_array), silence):

            # if a silence is detected within the search interval
            if max(numpy_array[i:i + silence]) < silent_threshold:

                # search back for the first 'loud' frame
                for j in range(i, i - silence, -1):

                    if abs(numpy_array[j]) > silent_threshold:
                        return global_peak_index + j

    def __get_beginnings(self):

        # detect melody beginnings
        beginnings = []
        for peak in self.__peaks:

            # in case the first peak is too close to the recording beginning
            start_ind = peak - rs.ROUGH_SEARCH
            if start_ind < 0:
                start_ind = 0

            beginning = self.__get_beginning(self.__rec_numpy_array[start_ind:peak+rs.ROUGH_SEARCH], peak, start_ind)

            beginnings.append(beginning)

        message = "{} peaks detected but only {} beginnings were detected".format(self.__peaks, len(beginnings))
        self.__assert(self.__actual_mel_quant == len(beginnings), message)

        return beginnings

    def __get_ends(self):

        ends = []
        for peak in self.__peaks:

            end = self.__get_end(self.__rec_numpy_array[peak:peak+rs.ROUGH_SEARCH], peak)

            ends.append(end)

        message = "{} peaks were detected but only {} ends are found".format(self.__actual_mel_quant, len(ends))
        self.__assert(self.__actual_mel_quant == len(ends), message)

        return ends

    def __clear_dir(self, path):

        if not self.__debug_mode:

            files_found = self.__get_files_from_dir(path)
            if files_found:

                for file_ in files_found:
                    os.remove('\\'.join(path + file_))

            self.__assert(len(self.__get_files_from_dir(path)) == 0, "Folder was not cleared")

    def __save_detected_melodies(self, path=rs.MELODY_PATH):
        """
        :param file_path_name: string with base file name. A number will be appended to
        each file when saving starting from 1. I may include path. Path should be
        separated with '\\' for Windows.
        :return:
        """

        self.__clear_dir(rs.MELODY_PATH)

        count = 1
        for beg, end in zip(self.__beginnings, self.__ends):

            timestamp = dt.now().strftime('%Y%m%d%H%M%S')
            file_name = rs.MELODY_FILE_NAME.format(timestamp + '_' + str(count))
            path_file = '\\'.join([path, file_name])
            write(path_file, rs.RATE, self.__rec_numpy_array[beg:end])

            dir_ = self.__get_files_from_dir(rs.MELODY_PATH)
            msg = "Melody {} wasn't exported".format(file_name)
            self.__assert(file_name in dir_, msg)
            count += 1

        file_qty = len(self.__get_files_from_dir(rs.MELODY_PATH))
        msg = "{} melodies were detected but only {} were exported".format(self.__actual_mel_quant, count-1)  # TODO after debug count-1 -> file_qty
        self.__assert(self.__actual_mel_quant == count-1, msg)  # TODO after debug count-1 -> file_qty

    def __get_repeats(self):
        repeats = []
        for beginning, end in zip(self.__beginnings, self.__ends):
            melody_number = self.__beginnings.index(beginning)+1
            rpt_quantity = self.__get_repeat_quantity(self.__rec_numpy_array[beginning:end], melody_number)
            repeats.append(rpt_quantity)
        return repeats

    def __get_melody_without_repeats(self):  # TODO probably not the best idea
        melody_without_repeats = None
        return melody_without_repeats

    def __get_melody_volumes(self):

        # TODO refactor to return volumes both in relation to media_volume, and without relation

        """
        We take the max volume of the 'loudest' melody as 100%
        This method works if the max volume of the 'loudest' melody really corresponds to 100% volume of the device
        :return:
        """

        # get absolute volumes of the melodies
        absolute_volumes = [max(self.__rec_numpy_array[beg:end]) for beg, end in zip(self.__beginnings, self.__ends)]

        # the max volume (will be equal to the current media volume)
        max_volume = max(absolute_volumes)

        # get melodies' volumes
        relative_volumes = [int(float(i) / max_volume * self.__media_volume) for i in absolute_volumes]

        message = "Invalid melody volume in {}".format(relative_volumes)
        self.__assert(all(0 < i <= 100 for i in relative_volumes), message)

        return relative_volumes

    def __set_debug_mode(self, debug):
        """
        :param debug: bool
        :return:
        """
        self.__debug_mode = debug

    @debugger
    def __print(self, *args):

        print(args)

    @debugger
    def __plot(self, data, title=''):

        plt.plot(data)
        plt.title(title)
        plt.show()

    @staticmethod
    def __get_files_from_dir(path):
        files = os.listdir(path)
        return files

    def __recognize_melodies(self, samples_path=rs.SAMPLES_PATH):

        analyzed_similarities = []
        samples = self.__get_files_from_dir(samples_path)

        for beg, end in zip(self.__beginnings, self.__ends):
            group_similarity = []

            for file_name in samples:
                sample = self.__read_from_wav_to_numpy('//'.join([samples_path, file_name]))
                melody = self.__rec_numpy_array[beg:end]

                similarity = self.__get_melody_similarity(sample, melody)

                melody_number = 'melody_{}'.format(self.__beginnings.index(beg)+1)
                group_similarity.append([melody_number, file_name, similarity])

            analyzed_similarities.append(group_similarity)

        return analyzed_similarities

    @staticmethod
    def __get_expected_melodies_qty(expected):
        expected_melodies_quantity = sum([len(i) for i in expected.values()])
        return expected_melodies_quantity

    @staticmethod
    def __create_melodies(recording, beginnings, ends):

        melodies = []

        for beg, end in zip(beginnings, ends):

            melodies.append(Melody(recording[beg:end]))

        return melodies

    def assure(self, expected, debug=False):
        """
        :return: True if all the detected data matched expected
        """
        self.__set_debug_mode(debug)

        self.__save_recording(rs.RECORD_PATH)

        self.__exp_mel_qty = self.__get_expected_melodies_qty(expected)

        # read .wav and store as numpy array for further analysis
        # self.__rec_numpy_array = self.__read_from_wav_to_numpy(rs.RECORD_PATH + self.__file) TODO uncomment after debug
        self.__rec_numpy_array = self.__read_from_wav_to_numpy('//'.join([rs.RECORD_PATH, '5.wav']))
        self.__print('all record length:', len(self.__rec_numpy_array))

        # self.__plot(self.__rec_numpy_array[::100], 'All the record')

        # find peaks
        self.__peaks = self.__get_peaks(self.__rec_numpy_array)
        self.__print('peaks: exp {} , actual {}'.format(self.__exp_mel_qty, len(self.__peaks)), self.__peaks)

        # find melodies beginnings
        self.__beginnings = self.__get_beginnings()
        if self.__debug_mode:
            self.__print('beginnings', self.__beginnings)

        # find melodies ends
        self.__ends = self.__get_ends()
        self.__print('ends', self.__ends)

        # creare Melody objects
        self.__melodies = self.__create_melodies(self.__rec_numpy_array, self.__beginnings, self.__ends)

        # TODO implement def save in Melody class

        # export the detected melodies
        self.__save_detected_melodies()

        # get melody trigger times in datetime format
        self.__melody_trigger_times = self.__get_melody_trigger_times()
        self.__print('trigger times', self.__melody_trigger_times)

        # TODO test this feature on a record via a cable with volume escalation from zero up
        # probably, it will bw better not to count repeats this way
        # but compare whole detected melodies with 'control' melodies
        # that will include melodies with repeats
        self.__repeats = self.__get_repeats()
        self.__print('repeats', self.__repeats)

        # for beg, end in zip(self.__beginnings, self.__ends):
        #     # if self.__beginnings.index(beg) == 18:
        #         plt.plot(self.__rec_numpy_array[beg:end])
        #         plt.title('Detected melody #{}'.format(self.__beginnings.index(beg) + 1))
        #         sd.play(self.__rec_numpy_array[beg:end], samplerate=rs.RATE)
        #         plt.show()

        # get volumes of the melodies
        self.__melody_volumes = self.__get_melody_volumes()  # TODO implement assertion
        self.__print('melody_volumes', self.__melody_volumes)

        # TODO test this feature on a record with volume escalation from zero up
        # self.__melodies_without_repeats = self.__get_melody_without_repeats()

        self.__recognized_melody_names = self.__recognize_melodies()
        print(self.__recognized_melody_names)
        #
        # # compare melody trigger times
        # trigger_times = self.__verify_trigger_times(expected)
        #
        # # compare melody repeat quantities
        # repeat_quantities = self.__verify_repeat_quantities(expected)
        #
        # # compare melody volumes
        # volumes = self.__verify_volumes(expected)
        #
        # # compare melody names
        # melody_names = self.__verify_melody_names(expected)
        #
        # return all([trigger_times, repeat_quantities, volumes, melody_names])

        return True



class Melody:
    def __init__(self):
        pass