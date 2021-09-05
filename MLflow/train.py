import os
import sys
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score

import mlflow as mlflow
import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from MLflow.prepare_data import read_imdb_data, prepare_imdb_data, build_dict, convert_and_pad_data, preprocess_data


class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()

        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0, :]
        reviews = x[1:, :]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())


def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch_X, batch_y = batch

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # TODO: Complete this train method to train the model provided.
            optimizer.zero_grad()
            output = model.forward(batch_X)

            loss = loss_fn(output, batch_y)

            loss.backward()

            optimizer.step()

            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))


def predict(input_data, model, word_dict):
    from prepare_data import convert_and_pad, review_to_words
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_X, data_len = convert_and_pad(word_dict, review_to_words(input_data))
    data_X, data_len = np.array(data_X), np.array(data_len)
    data_pack = np.hstack((data_len, data_X))
    data_pack = data_pack.reshape(1, -1)

    data = torch.from_numpy(data_pack)
    data = data.to(device)

    model.eval()

    result = torch.round(model(data)).item()

    return result


if __name__ == "__main__":
    #MLFLOW_SERVER_URL = 'http://192.168.1.105:8000/'
    #mlflow.set_tracking_uri(MLFLOW_SERVER_URL)
    #experiment_name = 'LSML2'
    #mlflow.set_experiment(experiment_name)

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    max_epochs = float(sys.argv[1]) if len(sys.argv) > 1 else 1
    batch_size = float(sys.argv[2]) if len(sys.argv) > 2 else 64
    learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
    embeding_dim = float(sys.argv[4]) if len(sys.argv) > 4 else 32
    hidden_dim = float(sys.argv[5]) if len(sys.argv) > 5 else 100
    vocab_size = float(sys.argv[6]) if len(sys.argv) > 6 else 5000

    data, labels = read_imdb_data(test_to_train_num=10000)
    train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
    train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)
    print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))

    word_dict = build_dict(train_X, vocab_size)

    train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
    test_X, test_X_len = convert_and_pad_data(word_dict, test_X)

    train_X = pd.concat([pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1)
    test_X = pd.concat([pd.DataFrame(test_X_len), pd.DataFrame(test_X)], axis=1)

    train_X = torch.from_numpy(np.array(train_X)).float().squeeze()
    test_X = torch.from_numpy(np.array(test_X)).float().squeeze()
    train_y = torch.from_numpy(np.array(train_y)).float().squeeze()
    test_y = torch.from_numpy(np.array(test_y)).float().squeeze()

    print(train_X.shape)
    print(train_y.shape)

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

    with mlflow.start_run():
        model = LSTMClassifier(embeding_dim, hidden_dim, vocab_size)

        torch.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(model.parameters())
        loss_fn = torch.nn.BCELoss()

        print("Using device {}.".format(device))
        print("Model loaded with batch_size {}, learning_rate {}, "
              "epochs {}, embedding_dim {}, hidden_dim {}, vocab_size {}.".format(
            batch_size, learning_rate, max_epochs, embeding_dim, hidden_dim, vocab_size
        ))

        train(model, train_loader, max_epochs, optimizer, loss_fn, device)

        predictions = predict(test_X, model, word_dict)
        accuracy = accuracy_score(test_y, predictions)

        print(" Accuracy: %s" % accuracy)

        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("embeding_dim", embeding_dim)
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("vocab_size", vocab_size)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.pytorch.log_model(model, "Classifier")
