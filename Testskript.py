import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image

st.header('This is an Image Captioning App')

@st.cache(ttl = 300)
def folder_assign():
    annotation_folder = '/annotations/'
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                cache_subdir=os.path.abspath('.'),
                origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
                , extract=True)
        return  os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
        os.remove(annotation_zip)
    else:
        return os.path.abspath('.') + '/annotations/captions_train2014.json'

        
annotation_file = folder_assign()
# Download image files
@st.cache(ttl = 300)
def download_pics():
    image_folder = '/train2014/'
    if not os.path.exists(os.path.abspath('.') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                cache_subdir=os.path.abspath('.'),
                origin='http://images.cocodataset.org/zips/train2014.zip'
                , extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
        return PATH
    else:
        return os.path.abspath('.') + image_folder


PATH = download_pics()

def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.keras.layers.Resizing(299, 299)(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path
    
    
# We will override the default standardization of TextVectorization to preserve
# "<>" characters, so we preserve the tokens for the <start> and <end>.    
def standardize(inputs):
      inputs = tf.strings.lower(inputs)
      return tf.strings.regex_replace(inputs,
                                  r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")
    
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


@st.cache(allow_output_mutation=True, ttl=1800)
def make_dictionary():
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
      caption = f"<start> {val['caption']} <end>"
      image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
      image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)
    train_image_paths = image_paths[:40000]
    
    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
      caption_list = image_path_to_caption[image_path]
      train_captions.extend(caption_list)
      img_name_vector.extend([image_path] * len(caption_list))
        
    ##dictionary ab hier
    caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

    # Max word count for a caption.
    max_length = 50
    # Use the top 5000 words for a vocabulary.
    vocabulary_size = 5000
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardize,
        output_sequence_length=max_length)
    # Learn the vocabulary from the caption data.
    tokenizer.adapt(caption_dataset)
    
    cap_vector = caption_dataset.map(lambda x: tokenizer(x))
    
    ##dicdionary bis hier
    return tokenizer

@st.cache(allow_output_mutation=True, ttl=1800, hash_funcs={"keras.utils.object_identity.ObjectIdentityDictionary": lambda _: None})
def generate_tokenizer() :
    return make_dictionary()
   

tokenizer = generate_tokenizer()
# Create mappings for words to indices and indices to words.
word_to_index = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())
index_to_word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)
# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
num_steps = 375 # // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


@st.cache(allow_output_mutation=True, ttl = 1800, hash_funcs={"keras.utils.object_identity.ObjectIdentityDictionary": lambda _: None,
                                                  "builtins.weakref": lambda _: None,
                                                  "tensorflow.python.training.tracking.base.TrackableReference": lambda _: None,  })
def check_checkpoints():
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    #if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

check_checkpoints()    

def evaluate(image):
    
    max_length = 50
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)



        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result

def Testmethode():   
    image_url = 'https://tensorflow.org/images/surf.jpg'
    image_extension = image_url[-4:]
    image_path = tf.keras.utils.get_file('image'+image_extension, origin=image_url)

    result = evaluate(image_path)
    st.write('Prediction Caption:', ' '.join(result))
    #plot_attention(image_path, result, attention_plot)
    # opening the image
    image = Image.open(image_path)
    st.image(image)
    
if st.button('Testknopf'):
     Testmethode()
