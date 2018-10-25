import torch
import torch.nn as nn

from parseMIDI import parseBeats, reconstructFromBeats, getTpb

class RNN(nn.Module):
    def __init__(self, input_size, output_size, batch_size = 1):
        super(RNN, self).__init__()

        self.output_size = output_size
        self.lstm = nn.LSTMCell(input_size, output_size)
        self.hx = torch.randn(batch_size, output_size)
        self.cx = torch.randn(batch_size, output_size)

    def forward(self, input):
        self.hx, self.cx = self.lstm(input, (self.hx, self.cx))
        return self.hx

    def reset(self):
        self.hx = torch.randn(batch_size, output_size)
        self.cx = torch.randn(batch_size, output_size)

def createMidiFile(filename, output, tpb):
    sequence = []
    for o in output:
        sequence += (o + 1).int().numpy().tolist()
    return reconstructFromBeats(filename, sequence, tpb)

def constructTensorFromMidi(filename):
    sequence = parseBeats(filename)
    return (torch.tensor(sequence) - 1).view(len(sequence), 1, 14).float()

rnn = RNN(14, 14)

filename = 'MIDI/melody/ashover1.mid'
# ashover1 = parseBeats('MIDI/melody/ashover1.mid')
tpb = getTpb('MIDI/melody/ashover1.mid')
# input = torch.tensor(ashover1).view(len(ashover1), 1, 14).float()


# output = []
# for i in range(len(input)):
#     output += [rnn(input[i])]
# sequence = constructTensorFromMidi(filename)
# print(output[0][0])
# print(sequence.long()[1][0])

# mid = createMidiFile('test.mid',output, tpb)

criterion = nn.MSELoss()
learning_rate = 0.005

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

def train(filenames, numEpochs = 10):
    output = []
    for f in filenames:
        sequence = constructTensorFromMidi(f)
        compSeq = sequence
        tpb = getTpb(f)
        for i in range(numEpochs):
            rnn.zero_grad()
            output = []
            for j in range(sequence.size()[0] - 1):
                output += [rnn(sequence[j])]
                loss = criterion(output[j][0], compSeq[j + 1][0])
                loss.backward(retain_graph = True)
                for p in rnn.parameters():
                    p.data.add_(-learning_rate, p.grad.data)
            print('Epoch: %d / %d, Elapsed time: %s' % (i + 1, numEpochs, timeSince(start)))
            mid = createMidiFile('f_epoch_%d.mid' % i, output, tpb)   
    mid = createMidiFile('f_final.mid',output, tpb)
    return mid

mid = train([filename])
for msg in mid.tracks[0]:
    print(msg)

# rnn = nn.LSTMCell(10, 20)
# input = torch.randn(6, 3, 10)
# hx = torch.randn(3, 20)
# cx = torch.randn(3, 20)
# output = []
# for i in range(6):
#         hx, cx = rnn(input[i], (hx, cx))
#         output.append(hx)

# print(output)