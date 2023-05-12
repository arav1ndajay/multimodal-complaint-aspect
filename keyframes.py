import cv2
import os


class GenerateKeyframes():
  def __init__(self):
    self.image_base_path = f"testframes/"

  def generateKeyframes(self, file_path: str, timestamps: list[list[int, int]], idx: int):

      image_path = self.image_base_path + str(idx)

      if not os.path.exists(image_path):
        os.makedirs(image_path)

      for tsmp in timestamps:
        vidcap = cv2.VideoCapture(file_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)

        start, end = tsmp

        success,image = vidcap.read()
        count = 0
        success = True

        timestamp_runner = -1

        while success:
          success,frame = vidcap.read()
          count+=1
          timestamp = int(count/fps)

          if timestamp >= start and timestamp <= end:
            if timestamp_runner == -1:
              timestamp_runner = timestamp
          
            # save a frame every 5 seconds in a timestamp interval
            if timestamp == timestamp_runner:
              cv2.imwrite(f"{image_path}/{timestamp_runner}-{min(timestamp_runner + 5, end)}.jpg", frame)
              timestamp_runner += 5
              if timestamp_runner >= end: break
          
          elif timestamp > end: break