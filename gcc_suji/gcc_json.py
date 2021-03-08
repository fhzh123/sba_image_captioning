import os
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from collections import OrderedDict
import json
import random
from PIL import Image
from scipy.misc import imread


data = pd.read_csv("/HDD/kyohoon/gcc/Train_GCC-training.tsv", delimiter='\t', header=None)
tokens = []

for i in data[0]:
    i = re.sub(' \'', '\'', i)
    i = re.sub(' - ', '-', i)
    tokens.append(text_to_word_sequence(i))

path_dir = '/HDD/kyohoon/gcc/train'
file_list = os.listdir(path_dir)

img_list = [int(i[:-4]) for i in file_list]
img_list = sorted(img_list)

for i in img_list:
    try :
        Image.open(path_dir+'/'+str(i)+'.jpg')
    except:
        img_list.remove(i)
        print(i, 'try_remove')
        continue
    image2 = imread(path_dir+'/'+str(i)+'.jpg')
    if len(image2.shape)<2:
        img_list.remove(i)
        print(i, 'imread_remove')
    image1 = Image.open(path_dir+'/'+str(i)+'.jpg')

    imag1_size = image1.size
    if imag1_size==(1, 1):
        img_list.remove(i)
        print(i, 'remove')

validtest_index = random.sample(img_list, int(len(img_list)*0.1))
test_index = random.sample(validtest_index, int(len(img_list)*0.1)//2)
valid_index = [i for i in validtest_index if i not in test_index]

print(len(test_index))
print(len(valid_index))

json_data = OrderedDict()
images = []
for i in img_list:
    img_name = str(i)+'.jpg'
    words = tokens[i]
    if i in valid_index:
        images.append({'file_name': img_name, 'tokens': words, 'split': 'valid'})
    elif i in test_index:
        images.append({'file_name': img_name, 'tokens': words, 'split': 'test'})
    else:
        images.append({'file_name': img_name, 'tokens': words, 'split': 'train'})
    if i%10000==0:
        print(i)

json_data['images'] = images

with open('/home/jangsj/dataset_gcc_split.json', 'w') as make_file:
    json.dump(json_data, make_file, ensure_ascii=False)