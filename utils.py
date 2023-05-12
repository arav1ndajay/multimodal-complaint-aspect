# convert time from MM:SS-MM:SS to [SS, SS]
def get_time_range_in_sec(time):
  # get rid of unintentional spaces
  time = time.replace(" ", "")

  # get starting and ending times
  start, end = time.split("-")

  # convert to seconds
  start = int(start.split(":")[0])*60 + int(start.split(":")[1])
  end = int(end.split(":")[0])*60 + int(end.split(":")[1])

  return [start, end]
  
# convert time from [SS, SS] to MM:SS-MM:SS
def get_time_range_in_format(time):

  start, end = time

  start = int(float(start))
  end = int(float(end))

  start = str(int(start/60)) + (":0" if int(int(start%60) / 10) == 0 else ":") +  str(int(start%60))
  
  end = str(int(end/60)) + (":0" if int(int(end%60) / 10) == 0 else ":") + str(int(end%60))
  
  return start + "-" + end