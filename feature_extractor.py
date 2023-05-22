import os 
import pickle

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


actors = os.listdir('celebrity_data')
# print(actors)

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join('celebrity_data', actor)):
        filenames.append(os.path.join('celebrity_data', actor, file))

# print(filenames)
# print(len(filenames))

# pickle.dump(filenames, open('filenames.pkl', 'wb'))

filenames = pickle.load(open('filenames.pkl', 'rb'))

model = VGGFace(model= 'resnet50', include_top=False, input_shape=(224,224,3), pooling ='avg')

# print(model.summary)

def feature_extractor(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))
    # result = feature_extractor(file, model)
    # print(result.shape)
    # break


pickle.dump(features,open('embedding.pkl','wb'))



