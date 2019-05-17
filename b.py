from PIL import Image
for i in range(501,502):
    img=Image.open("/home/darshanpc/Documents/brain-tumor/703.png".format(i))
    img=img.resize((32,32),Image.ANTIALIAS)
    img.save("/home/darshanpc/Documents/brain-tumor/1.png".format(i))
    print(i)
