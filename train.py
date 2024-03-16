import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import math
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from attention import EncoderRNN, AttnDecoderRNN
from filter_train import filterPairs
from lang import Lang
from io import open


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1


# setting up MAX_LENGTH to get rid of long sentences
MAX_LENGTH = 30


# filter and separate data by vocabularies
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# reading pairs and creating vocabularies
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    pairs = []

    with open("text.ru-filtered.ru.subword", "r", encoding="utf-8") as file_1:
        lines_ru = file_1.readlines()

    with open("text.ba-filtered.ba.subword", "r", encoding="utf-8") as file_2:
        lines_ba = file_2.readlines()

    for ru, ba in zip(lines_ru, lines_ba):
        pairs.append([ru, ba])

    pairs = pairs[:800000]  # limit if need

    for i in range(len(pairs)):
        if "\n" in pairs[i][0] and "\n" in pairs[i][1]:
            pairs[i][0] = pairs[i][0][:-3]
            pairs[i][1] = pairs[i][1][:-3]

            pairs[i][0] += "."
            pairs[i][1] += "."

    # reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData("ru", "ba", False)
print(random.choice(pairs))


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# preparing data loader to train model
def get_dataloader(batch_size, reverse):

    if reverse:
        input_lang, output_lang, pairs = prepareData("ru", "ba", True)
    else:
        input_lang, output_lang, pairs = prepareData("ru", "ba", False)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, : len(inp_ids)] = inp_ids
        target_ids[idx, : len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(
        torch.LongTensor(input_ids).to(device), torch.LongTensor(target_ids).to(device)
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size
    )
    return input_lang, output_lang, train_dataloader


def train_epoch(
    dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))


def train(
    train_dataloader,
    encoder,
    decoder,
    n_epochs,
    learning_rate=0.001,
    print_every=100,
    plot_every=100,
    resume=False,
):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # reset every print_every
    plot_loss_total = 0  # reset every plot_every

    # define optimizers and criterion
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    start_epoch = 90

    # check if resuming training
    if resume:
        # load saved checkpoint
        checkpoint = torch.load("load\\checkpoint_90_epoch.pth")
        start_epoch = checkpoint["epoch"]
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])
        decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer_state_dict"])
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
        plot_losses = checkpoint["plot_losses"]
        print_loss_total = checkpoint["print_loss_total"]
        plot_loss_total = checkpoint["plot_loss_total"]
        start = checkpoint["start"]

    for epoch in range(start_epoch + 1, n_epochs + 1):
        loss = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += loss
        plot_loss_total += loss

        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print(
            "%s (%d %d%%) %.4f"
            % (
                timeSince(start, epoch / n_epochs),
                epoch,
                epoch / n_epochs * 100,
                print_loss_avg,
            )
        )

        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

        # save checkpoint after each epoch
        torch.save(
            {
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
                "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "plot_losses": plot_losses,
                "print_loss_total": print_loss_total,
                "plot_loss_total": plot_loss_total,
                "start": start,
            },
            "load\\checkpoint_" + str(epoch) + "_epoch.pth",
        )


hidden_size = 128
batch_size = 64

input_lang, output_lang, train_dataloader = get_dataloader(batch_size, False)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)


train(
    train_dataloader,
    encoder,
    decoder,
    n_epochs=100,
    learning_rate=0.001,
    print_every=100,
    plot_every=100,
    resume=False,
)
