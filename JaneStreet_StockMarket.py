from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate, Lambda, GaussianNoise, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np
import pandas as pd
from tqdm import tqdm
from random import choices


SEED = 1111

np.random.seed(SEED)

train = pd.read_csv('../input/jane-street-market-prediction/train.csv')
train = train.query('date > 85').reset_index(drop = True) 
train = train[train['weight'] != 0]

train.fillna(train.mean(),inplace=True)

train['action'] = ((train['resp'].values) > 0).astype(int)


features = [c for c in train.columns if "feature" in c]

f_mean = np.mean(train[features[1:]].values,axis=0)

resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']

X_train = train.loc[:, train.columns.str.contains('feature')]
#y_train = (train.loc[:, 'action'])

y_train = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T

# fit
def create_mlp(
    num_columns, num_labels, hidden_units, dropout_rates, label_smoothing, learning_rate
):

    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)
    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 1])(x)
    
    x = tf.keras.layers.Dense(num_labels)(x)
    out = tf.keras.layers.Activation("sigmoid")(x)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tfa.optimizers.RectifiedAdam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=tf.keras.metrics.AUC(name="AUC"),
    )

    return model

epochs = 200
batch_size = 4096
hidden_units = [160, 160, 160]
dropout_rates = [0.2, 0.2, 0.2, 0.2]
label_smoothing = 1e-2
learning_rate = 1e-3

tf.keras.backend.clear_session()
tf.random.set_seed(SEED)
clf = create_mlp(
    len(features), 5, hidden_units, dropout_rates, label_smoothing, learning_rate
    )

clf.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

# save model
clf.save(f'model.h5')

# inference
th = 0.502
models = [clf]
f = np.median
import janestreet
env = janestreet.make_env()
for (test_df, pred_df) in tqdm(env.iter_test()):
    if test_df['weight'].item() > 0:
        x_tt = test_df.loc[:, features].values
        if np.isnan(x_tt[:, 1:].sum()):
            x_tt[:, 1:] = np.nan_to_num(x_tt[:, 1:]) + np.isnan(x_tt[:, 1:]) * f_mean
        pred = np.mean([model(x_tt, training = False).numpy() for model in models],axis=0)
        pred = f(pred)
        pred_df.action = np.where(pred >= th, 1, 0).astype(int)
    else:
        pred_df.action = 0
    env.predict(pred_df)