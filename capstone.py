import streamlit as st

logo1 = st.image("logo.PNG")
st.logo("logo.png")

st.title(f"Search for Images by Captions")
st.subheader(f"Dhir's Disciples")

caption = st.text_input('Type in a caption query...')

st.text("Here are 4 images that are similar to your query:")
urls = ["https://img-cdn.pixlr.com/image-generator/history/65bb506dcb310754719cf81f/ede935de-1138-4f66-8ed7-44bd16efc709/medium.webp", "https://img-cdn.pixlr.com/image-generator/history/65bb506dcb310754719cf81f/ede935de-1138-4f66-8ed7-44bd16efc709/medium.webp", "https://img-cdn.pixlr.com/image-generator/history/65bb506dcb310754719cf81f/ede935de-1138-4f66-8ed7-44bd16efc709/medium.webp", "https://img-cdn.pixlr.com/image-generator/history/65bb506dcb310754719cf81f/ede935de-1138-4f66-8ed7-44bd16efc709/medium.webp"]

##############################################
from cogworks_data.language import get_data_path
import numpy as np
from pathlib import Path
import json
from collections import Counter

# load COCO metadata
filename = get_data_path("captions_train2014.json")
with Path(filename).open() as f:
    coco_data = json.load(f)
    
import re, string
punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
from gensim.models import KeyedVectors
filename = "glove.6B.200d.txt.w2v"

# this takes a while to load -- keep this in mind when designing your capstone project
glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)

punc_regex = re.compile(f'[{re.escape(string.punctuation)}]')

filename = "glove.6B.200d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)

def preprocess_captions(captions):
    words = []
    for caption in captions:
        cleaned_caption = punc_regex.sub('', caption).lower()
        words2 = cleaned_caption.split()
        words.extend(words2)

    word_counts = Counter(words)
    num_docs = len(captions)
    weights = {word: np.log10(num_docs / count) for word, count in word_counts.items()}
    return weights
    
all_captions = [caption_info["caption"] for caption_info in coco_data["annotations"]]
ids = [caption_info["image_id"] for caption_info in coco_data["annotations"]]
idf_weights = preprocess_captions(all_captions)

def embed_caption(caption, dim=200):
    
    cleanCap = punc_regex.sub('', caption).lower()
    tokens = cleanCap.split()

    real_embeddings = []
    for word in tokens:
        
        if word in glove:
            idf = idf_weights.get(word)
            
            real_embeddings.append(idf * glove[word])  # Directly calculate weighted embedding
        else:
            real_embeddings.append(np.zeros(dim))  # Zero vector for missing words
    if not real_embeddings:  #No words
        return np.zeros(dim)
    
    # Sum
    caption_embedding = np.sum(real_embeddings, axis=0)
    normalized_embedding = caption_embedding / np.linalg.norm(caption_embedding)  # Normalize
    return normalized_embedding

all_embeddings = [embed_caption(caption) for caption in all_captions]

final_embeddings = {}

for i in np.arange(len(all_embeddings)):
    final_embeddings[ids[i]] = all_embeddings[i]

# Load saved image descriptor vectors
import pickle
from cogworks_data.language import get_data_path

from pathlib import Path

with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
    resnet18_features = pickle.load(f)

#print(resnet18_features.keys())

import mygrad as mg
import mynn
import numpy as np

from mygrad.nnet.initializers import he_normal
from mygrad.nnet import margin_ranking_loss
from mynn.layers.dense import dense
from mynn.losses.mean_squared_loss import mean_squared_loss
from mynn.optimizers.sgd import SGD
from mynn.optimizers.adam import Adam

import matplotlib.pyplot as plt

from mygrad.nnet.initializers.he_normal import he_normal

class Model:
    def __init__(self):
        """ This initializes all of the layers in our model, and sets them
        as attributes of the model.
       
        """
        self.M = he_normal(512, 200)
        self.b = he_normal(1, 200)
        
    def __call__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        '''
        return x @ self.M + self.b
        
    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters`
        
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
       
        return (self.M, self.b)
        
from cogworks_data.language import get_data_path

from pathlib import Path
import json

# load COCO metadata
filename = get_data_path("captions_train2014.json")
with Path(filename).open() as f:
    coco_data = json.load(f)
    
#set(resnet18_features) < set(img["id"] for img in coco_data["images"])

#randomly shuffle?
split = int(82612*.8)

training = list(set(img["image_id"] for img in coco_data["annotations"]))[:split]
validation = list(set(img["image_id"] for img in coco_data["annotations"]))[split:]

from cogworks_data.language import get_data_path
from pathlib import Path
import json
from collections import defaultdict
#!pip install image_search
import image_search

train = []
skipped = []
descriptors1 = []
descriptors2 = []
captions = []

import random

for id in training:
     
    confusor_image = None
    
    while confusor_image is None:
        try:
            confusor_image = resnet18_features[training[random.randrange(split - 1)]]
        except:
            continue
    try:
        if resnet18_features[id] is None:
            continue
        else:
            #database.get_caption(id)
            train.append([final_embeddings.get(id), resnet18_features[id], confusor_image])
            descriptors1.append(resnet18_features[id])
            descriptors2.append(confusor_image)
            captions.append(final_embeddings.get(id))

            if resnet18_features[id].shape != (1,512):
                print(id)
            if confusor_image.shape != (1,512):
                print(confusor_image)
   
    except:
        skipped.append(id)

#print(len(skipped))
#print(len(train))
train = np.asarray(train, dtype = "object")
#print(train[0])

# STUDENT CODE HERE
import mygrad as mg

model = Model()

optim = SGD(model.parameters, learning_rate=0.1, momentum=0.9)  #
batch_size = 32
#65955
for epoch_cnt in range(1):
    idxs = np.arange(32)
    np.random.shuffle(idxs)
    
    for batch_cnt in range(0, 32//batch_size):
        batch_indices = idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
       
        batch = np.asarray(descriptors1)[batch_indices.astype(int)]  # random batch of our training data
        #print(batch.shape)
        # `model.__call__ is responsible for performing the "forward-pass"
        prediction = model(batch)
        
        #loss = margin_ranking_loss(descriptors1[batch_indices[0]], descriptors2[batch_indices[0]], captions[batch_indices[0]], 0.1)
        loss = mg.mean(mg.maximum(0, 0.1 - np.asarray(captions)[batch_indices.astype(int)] * (prediction - model(np.asarray(descriptors2)[batch_indices.astype(int)]))))
        # you still must compute all the gradients!
        loss.backward()
        
        # the optimizer is responsible for updating all of the parameters
        optim.step()
        

import pickle

# save the iris classification model as a pickle file
model_pkl_file = "model.pkl"

with open(model_pkl_file, 'wb') as file:
    pickle.dump(model, file)

with open(model_pkl_file, 'rb') as file:
    model = pickle.load(file)

image_database = {}
for id in training:
    try:
        if resnet18_features[id] is not None:
            descriptor = resnet18_features[id]
            image_databse[id] = model(descriptor)
    except:
        continue

captions = [ann['caption'] for ann in coco_data['annotations']]
all_words = [word for caption in captions for word in caption.split()]
word_counts = Counter(all_words)
num_docs = len(captions)
idf = {}
for word, count in word_counts.items():
    idf[word] = np.log(num_docs / (count + 1))

def embed_caption(caption, glove, idf):
    tokens = [punc_regex.sub('', w.lower()) for w in caption.split()] # Remove punctuation
    real_embeddings = []
    for word in tokens:
        if word in glove:
            embedding = glove[word]
            weight = idf.get(word, 0)
            real_embeddings.append(embedding * weight)
    if real_embeddings:
        embedding_sum = np.sum(real_embeddings, axis=0)
        return embedding_sum / np.linalg.norm(embedding_sum)
    else:
        return np.zeros(200)


all_captions = [caption_info["caption"] for caption_info in coco_data["annotations"]]

all_embeddings = [embed_caption(caption,glove,idf) for caption in all_captions]

final_embeddings = {}
ids = [caption_info["image_id"] for caption_info in coco_data["annotations"]]

for i in np.arange(len(all_embeddings)):
    final_embeddings[ids[i]] = all_embeddings[i][1]


def query(q, glove, idf, caption, k=5):
    #weights = preprocess_captions([caption])
    #print(caption)
    #caption_embedding = embed_caption(weights, caption)
    
    #caption_embedding = glove[caption]
    caption_embedding = embed_caption(q, glove, idf)
    
    similarities = []
    for image_id, image_embedding in image_database.items():
        similarity_score = np.dot(caption_embedding, image_embedding)
        similarities.append((image_id, similarity_score))
    
    similarities.sort()
    
    return similarities[:k]
    
#print(query('sunny day'))

queried = query(caption,glove,idf, caption)

image2url = {image["id"]: image["coco_url"] for image in coco_data["images"]}

#queried = [57870, 384029, 222016, 520950]

urls = [image2url.get(id) for id in queried]

if len(urls) == 0:
    st.text("Done")

##############################################

for url in urls:
    st.image(url,width=400)

