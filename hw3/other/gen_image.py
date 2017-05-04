from PIL import Image
from PIL import ImageChops
import csv
import sys

fileX = "train.csv"
path = "pic/"

data_raw = list(csv.reader(open(fileX,'r')))
x_data = []
y_data = []

print ("Begin loading...")
for item in data_raw[1:]:
        y_data.append(int(item[0]))
        x_data.append([int(i) for i in item[1].split(" ")])
print ("Loading completed")

for count in range(0,len(x_data)):
        im = Image.new('L', (48,48))
        im.putdata(x_data[count])
        im.save(path+str(count)+"_"+str(y_data[count])+".jpg")
