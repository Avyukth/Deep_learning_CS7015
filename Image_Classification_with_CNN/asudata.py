import numpy as np
import pandas as pd
from augutil import Hflip,Vflip,Rrot,Rtra
from multiprocessing import Pool
from PIL import Image, ImageOps
pool = Pool()

def data_augmentation(i,train_x):


    im_array=train_x[i][:784].reshape(28,28)
    im_image=Image.fromarray(im_array.astype("uint8"))
    out_1=pool.apply_async(Hflip, [im_image]) 
    out_2=pool.apply_async(Vflip, [im_image]) 
    out_3=pool.apply_async(Rrot, [im_image]) 
    out_4=pool.apply_async(Rtra, [im_image]) 
    out_H=out_1.get()
    out_V=out_2.get()
    out_R=out_3.get()
    out_T=out_4.get()
    oti_img=im_array.reshape(1,784)
    
    out_stack_new=np.vstack((oti_img,out_H,out_V,out_R,out_T))
    out_label_new=np.vstack(([train_x[i][-1]]*5))
    print(i)

        

    return out_stack_new,out_label_new

    
train = pd.read_csv("data/train.csv")
train_x = np.array(train.drop("id", axis=1))

for i in range(train_x.shape[0]):
    out_stack_old,out_label_old=data_augmentation(i,train_x)
    data_id = np.concatenate((out_stack_old,out_label_old), axis=1)
    tag=list(train)[1:]
    data_df = pd.DataFrame(data_id, columns=tag)

    f= open('augment_data.csv', 'a')
    if i ==0:
        data_df.to_csv(f, sep=',', encoding='utf-8', index=False)
    else:
        data_df.to_csv(f, sep=',', encoding='utf-8', index=False,header=False)
    f.close()

