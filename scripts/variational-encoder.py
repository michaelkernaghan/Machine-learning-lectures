import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = '/home/mike/Machine-learning-lectures/data/google-ethereum-transactions.csv'
data = pd.read_csv(file_path)

# Fill missing values
data.fillna(0, inplace=True)

# Convert block_timestamp to datetime
data['block_timestamp'] = pd.to_datetime(data['block_timestamp'])

# Extract time-based features
data['hour'] = data['block_timestamp'].dt.hour
data['day_of_week'] = data['block_timestamp'].dt.dayofweek
data['day_of_month'] = data['block_timestamp'].dt.day

# Calculate gas usage efficiency
data['gas_usage_efficiency'] = np.where(data['gas'] != 0, data['value'] / data['gas'], 0)

# Calculate gas price to value ratio
data['gas_price_value_ratio'] = np.where(data['value'] != 0, data['gas_price'] / data['value'], 0)

# Calculate address frequency features
data['from_address_frequency'] = data.groupby('from_address')['from_address'].transform('count')
data['to_address_frequency'] = data.groupby('to_address')['to_address'].transform('count')

# Define features
features = ['gas_price', 'value', 'gas', 'transaction_type', 'hour', 'day_of_week', 'day_of_month', 
            'gas_usage_efficiency', 'gas_price_value_ratio', 'from_address_frequency', 'to_address_frequency']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Split the data
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Define the VAE model
original_dim = X_train.shape[1]
input_shape = (original_dim, )
intermediate_dim = 64
latent_dim = 2

inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

# Train the VAE model
vae.fit(X_train, epochs=50, batch_size=32, validation_data=(X_test, None))

# Generate new transactions
z_sample = np.array([[0, 0]])
generated_data = decoder.predict(z_sample)
print("Generated Data:", scaler.inverse_transform(generated_data))
