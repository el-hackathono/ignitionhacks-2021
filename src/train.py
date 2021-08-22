"""AI trainer."""

# credit:
#   Andrew Shao's example

import string

# pylint: disable=E0401
import torch
from torch import nn, optim
from torch.nn import functional as fnl
from torch.utils import data

import torchtext
import numpy as np
import pandas
import nltk
from transformers import AutoTokenizer, BertModel, BertConfig

TEST_CUTOFF = 1000
BATCH_SIZE = 128
EPOCHS = 50

def main():
    """Main thread."""
    nltk.download("punkt")

    train, test = torchtext.datasets.YelpReviewPolarity(
        root=".data",
        split=("train", "test")
    )

    samples, results = load(train)
    samples = scrub(samples[:TEST_CUTOFF])

    tsamples, tresults = load(test)
    tsamples = scrub(tsamples[:TEST_CUTOFF])

    trainer = Trainer(Model(len(WORD2IDX), len(WORD2IDX) - 1))
    trainer.run(
        EPOCHS,
        get_dataloader(samples, results, TEST_CUTOFF, BATCH_SIZE),
        get_dataloader(tsamples, tresults, TEST_CUTOFF, BATCH_SIZE)
    )
    trainer.save()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
WORD2IDX = TOKENIZER.vocab
IDX2WORD = {WORD2IDX[word]: word for word in WORD2IDX}
PAD_TOKEN = TOKENIZER.pad_token_id
UNK_TOKEN = TOKENIZER.unk_token_id
MAX_LEN = 64


def tokenize(x):
    """Tokenize samples."""
    x = x.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokenized_x = nltk.word_tokenize(x)

    new_sent = [] 
    for word in tokenized_x:
        word = word.lower()
        if word not in WORD2IDX:
            new_sent.append(UNK_TOKEN)
        else:
            new_sent.append(WORD2IDX[word])

    new_sent = new_sent[:MAX_LEN]
    # Pad up to max length 
    padded_sent = [PAD_TOKEN] * MAX_LEN
    padded_sent[:len(new_sent)] = new_sent
    return padded_sent


def load(data):
    """
    Gather test cases and expected results.

    1 = neg, 2 = pos.
    """
    results, samples = [], []
    for y, x in data:
        samples.append(x)
        results.append(y)
    return samples, results


def scrub(samples):
    """Tokenizes samples and returns a new set."""
    new_samples = []
    for sample in samples:
        new_samples.append(tokenize(sample))
    return new_samples


class Dataset(data.Dataset):
    """X Y pair."""

    def __init__(self, tokenized_x, y):
        """Initialize x and y."""
        self.x = tokenized_x
        self.y = y

    def __len__(self):
        """Return length."""
        return len(self.x)

    def __getitem__(self, i):
        """Return item at position."""
        x = self.x[i]
        # offset range from {1, 2} to {0, 1}
        y = self.y[i] - 1
        return np.array(x), np.array(y)


def get_dataloader(samples, results, length, batch_size):
    """Create a `Datalaoder` object for samples."""
    dataset = Dataset(samples[:length], results[:length])
    return data.DataLoader(dataset, batch_size, shuffle=False)


class Model(nn.Module):
    """Neural network."""

    def __init__(self, vocab_len, padding_idx):
        """Construct RNN neural network."""
        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_len = vocab_len
        self.embedding_dim = 50
        self.hidden_dim = 16
        self.num_layers = 1

        self.embedding_layer = nn.Embedding(self.vocab_len, self.embedding_dim, padding_idx=self.padding_idx)
        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(self.hidden_dim * 2 * self.num_layers, 1)

    def forward(self, x):
        """Forward pass function."""
        embedding = self.embedding_layer(x)
        _, (hidden, cell) = self.rnn(embedding)  

        # hidden - The final state we use 
        # hidden: Tensor(Num layers, N, Dimensional) 
        hidden = hidden.transpose(0, 1)
        B = hidden.shape[0]

        hidden = hidden.reshape(B, -1)
        # classifier head
        logits = self.classifier(hidden)
        logits = logits.reshape(-1,)
        return logits


class Trainer:
    """Neural network trainer."""

    def __init__(self, model):
        """Initialize trainer model with loss function and optimizer."""
        self.device = DEVICE
        self.model = model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.loss_function = nn.BCEWithLogitsLoss()

    def save(self):
        """Save state of network."""
        torch.save(self.model.state_dict(), ".data/model.pth")

    def training_step(self, x, y):
        """Train with a single input."""
        self.model.train()

        # flush gradient
        self.optimizer.zero_grad()

        # forward pass
        outputs = self.model.forward(x)

        # loss function
        loss = self.loss_function(outputs, y)

        # backward pass
        loss.backward()

        self.optimizer.step()

    def train(self, train_dataloader):
        """Single training iteration for entire sample set."""
        for x, y in train_dataloader:
            x = x.long().to(self.device)
            y = y.float().to(self.device)
            self.training_step(x, y)

    def eval_step(self, x, y):
        """Evaluate a single input."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)

        loss = self.loss_function(output, y)

        output = torch.round(torch.sigmoid(output))

        accuracy = output == y
        tp = torch.sum(accuracy)
        all_ = accuracy.reshape(-1).shape[0]
        return loss, tp / all_

    def eval(self, eval_dataloader):
        """Calculate net loss and accuracy across samples."""
        net_loss = 0
        net_accuracy = 0
        n = 0

        for x, y in eval_dataloader:
            x = x.long().to(self.device)
            y = y.float().to(self.device)

            loss, accuracy = self.eval_step(x, y)
            net_loss += loss
            net_accuracy += accuracy
            n += 1

        net_loss /= n
        net_accuracy /= n
        print(f"Loss: {net_loss}; Accuracy: {net_accuracy}")

    def run(self, num_epochs, train_dataloader, eval_dataloader):
        """Train and evaluate a single epoch."""
        for epoch in range(num_epochs):
            self.train(train_dataloader)
            print(f"[{epoch}]: ", end="")
            self.eval(eval_dataloader)


if __name__ == "__main__":
    main()
