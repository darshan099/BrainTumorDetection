
import h5py
import numpy as np
from PIL import Image

for i in range(2854,3065):
    with h5py.File('dataset4/{}.mat'.format(i),'r') as file:    
        a=np.array(file['cjdata']['image'])
    im=Image.fromarray(a)    
    im.save('/home/darshanpc/Documents/brain-tumor/data/training_data/4/{}.png'.format(i))
    print(i)
