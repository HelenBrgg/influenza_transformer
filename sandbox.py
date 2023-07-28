"""
Showing how to use the model with some time series data.

NB! This is not a full training loop. You have to write the training loop yourself.

I.e. this code is just a starting point to show you how to initialize the model and provide its inputs

If you do not know how to train a PyTorch model, it is too soon for you to dive into transformers imo :)

You're better off starting off with some simpler architectures, e.g. a simple feed forward network, in order to learn the basics
"""

import dataset as ds
import utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import datetime
import transformer_timeseries as tst
import numpy as np
from tqdm import tqdm


# Hyperparams
test_size = 0.1
batch_size = 48
target_col_name = "FCR_N_PriceEUR"  # "elevation_profile2"
timestamp_col = "timestamp"  # "Zeit" #
# Only use data from this date and onwards
cutoff_date = datetime.datetime(2017, 1, 1)

# Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92  # length of input given to decoder
enc_seq_len = 153  # length of input given to encoder
# target sequence length. If hourly data and length = 48, you predict 2 days ahead
output_sequence_length = 48
# used to slice data into sub-sequences
window_size = enc_seq_len + output_sequence_length
step_size = 1  # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False

# Define input variables
exogenous_vars = []  # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0  # index position of target in batched trg_y

input_size = len(input_variables)

# Read data
data = utils.read_data(timestamp_col_name=timestamp_col)

# Remove test data from dataset
training_data = data[:-(round(len(data)*test_size))]

# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc.
# Should be training data indices only
training_indices = utils.get_indices_entire_sequence(
    data=training_data,
    window_size=window_size,
    step_size=step_size)

# Making instance of custom dataset class
training_data = ds.TransformerDataset(
    data=torch.tensor(training_data[input_variables].values).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    dec_seq_len=dec_seq_len,
    target_seq_len=output_sequence_length
)

# Making dataloader
training_data = DataLoader(training_data, batch_size)

# i, batch = next(enumerate(training_data))

# src, trg, trg_y = batch


# Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]


model = tst.TimeSeriesTransformer(
    input_size=len(input_variables),
    dec_seq_len=enc_seq_len,
    batch_first=batch_first,
    num_predicted_features=1
)


# output = model(
#   src=src,
#  tgt=trg,
# src_mask=src_mask,
# tgt_mask=tgt_mask
# )
# Loss and Optimizer
learning_rate = 0.01
n_epochs = 1000
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-9)

# Make src mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, enc_seq_len]
src_mask = utils.generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=enc_seq_len
)

# Make tgt mask for decoder with size:
# [batch_size*n_heads, output_sequence_length, output_sequence_length]
tgt_mask = utils.generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=output_sequence_length
)

# Training Loop
for epoch in range(n_epochs):
    for i, (src, trg, tgt_y) in enumerate(training_data):
        if batch_first == False:

            # shape_before = src.shape
            src = src.permute(1, 0, 2)
            # print("src shape changed from {} to {}".format(
            #   shape_before, src.shape))

            # shape_before = trg.shape
            trg = trg.permute(1, 0, 2)
            # print("src shape changed from {} to {}".format(
            #   shape_before, src.shape))

        # Make forecasts
        prediction = model(src, trg, src_mask, tgt_mask)
        # tgt_y = tgt_y.unsqueeze(1)

        # Compute and backprop loss
        loss = loss_fn(tgt_y, prediction)

        loss.backward()

        # Take optimizer step
        optimizer.step()

        optimizer.zero_grad()

        """ # Make forecasts
    y_predicted = model(
        src=src,
        tgt=trg,
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )  # ??=output?

    # Compute and backprop loss
    l = loss_fn(trg_y, y_predicted)

    l.backward()

    # Take optimizer step
    optimizer.step()

    # predcit = forward pass with our model

    # loss
   # l = loss_fn(trg_y, y_predicted)

    # calculategradient = backward_pass
    # l.backward()

    # update weights
   # optimizer.step()

    # zero the gradients after updating
    # optimizer.zero_grad(set_to_none=True)"""

        if (epoch+1) % 10 == 0:
            w, b = model.parameters()  # unpack parameters
            print('epoch', epoch+1, ' : w=',
                  w[0][0].item(), 'loss= ', loss.item())
    # print(
        # f'Prediction after training: f({X_test.item()})={model(X_test).item():.3f}')
