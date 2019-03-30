import recognition


def record(minutes):
    """
    Flow example: record_1 = record(60)
                  record_1.assure

    :param minutes: recording duration in minutes
    :return: an object containing the record
    """
    # current_media_volume = gm_app.get_media_volume()
    current_media_volume = 36  # TODO remove after debug

    return recognition.RecordingObject(minutes, current_media_volume)
