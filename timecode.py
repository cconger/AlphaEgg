import re

def seconds(timecode):
  match = re.match(r'(\d\d):(\d\d\.\d\d)', timecode)
  if match == None:
    return -1.0
  (minutes, seconds) = match.group(1,2)

  return (float(minutes) * 60) + float(seconds)