from PIL import Image

filePath1 = "lena.png"
filePath2 = "lena_modified.png"
im = Image.open(filePath1)
im2 = Image.open(filePath2)
imSize = im.size
newImage = Image.new("RGBA", imSize)
for x in range(0,imSize[0]):
	for y in range(0,imSize[1]):
		rgb1 = im.getpixel((x,y))
		rgb2 = im2.getpixel((x,y))
		if(rgb1 != rgb2):
			newImage.putpixel((x,y), rgb2)	

newImage.save("ans_two.png")