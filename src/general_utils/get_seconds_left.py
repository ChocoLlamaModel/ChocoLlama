import subprocess
import os

def get_seconds_left():
    command = [
        'squeue',
        '-h',
        '-j',
        os.environ['SLURM_JOB_ID'],
        '-O',
        'TimeLeft'
    ]
    
    time_left_string = subprocess.check_output(command).decode('utf-8')
    days_hours_minutes_seconds_split = time_left_string.split('-')
    days_string = None
    if len(days_hours_minutes_seconds_split) == 1:
        hours_minutes_seconds_string = days_hours_minutes_seconds_split[0]
    elif len(days_hours_minutes_seconds_split) == 2:
        days_string, hours_minutes_seconds_string = days_hours_minutes_seconds_split
    else:
        raise RuntimeError(f'Could not parse {time_left_string}')

    hours_minutes_seconds_split = hours_minutes_seconds_string.split(':')
    hours_string = None
    minutes_string = None
    seconds_string = None
    if len(hours_minutes_seconds_split) == 3:
        hours_string, minutes_string, seconds_string = hours_minutes_seconds_split
    elif len(hours_minutes_seconds_split) == 2:
        minutes_string, seconds_string = hours_minutes_seconds_split
    elif len(hours_minutes_seconds_split) == 1:
        seconds_string = hours_minutes_seconds_split[0]
    else:
        raise RuntimeError(f'Could not parse {time_left_string}')

    time_left = 0
    for string, in_seconds in ((days_string, 24*60*60), 
                            (hours_string,60*60),
                            (minutes_string,60),
                            (seconds_string,1)):
        if string is not None:
            time_left += in_seconds*int(string)

    return time_left





