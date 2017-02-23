from PIL import Image

filePath1 = "Q2 testing data/ans_Vienna.png"
filePath2 = "test.png"
im = Image.open(filePath1)
im2 = Image.open(filePath2)

imSize = im.size
print (imSize[0])
flag = True
for x in range(0,imSize[0]):
	for y in range(0,imSize[1]):
		rgb1 = im.getpixel((x,y))
		rgb2 = im2.getpixel((x,y))
		if(rgb1 != rgb2):
			flag = False

print(flag)