import torch
import torch.nn as nn
import torch.optim as optim

from parseMIDI import parseBeats, parseNotes, reconstructFromBeats, reconstructFromNotes, reconstruct, getTpb

import glob

def findFiles(path): return glob.glob(path)

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

def createMidiFileBeats(filename, output, tpb):
    sequence = []
    for o in output:
        sequence += (o + 1).int().numpy().tolist()
    return reconstructFromBeats(filename, sequence, tpb)

def createMidiFileNotes(filename, output, tpb):
    sequence = []
    for out in output:
        o = out[0]
        note = [i for i, j in enumerate(o[:12]) if j == max(o[:12])][0]
        pitch = [i for i, j in enumerate(o[12:22]) if j == max(o[12:22])][0]
        length = [i for i, j in enumerate(o[22:]) if j == max(o[22:])][0]

        """Uncomment print statements to view output sequence"""
        # print('Sequence for %s' % filename)
        # print((pitch*12 + note, 90, length))

        sequence += [(pitch*12 + note, 90, length)]
    # print('\n')
    return reconstruct(filename, sequence, tpb)

def constructTensorFromMidiBeats(filename):
    sequence = parseBeats(filename)
    return (torch.tensor(sequence) - 1).view(len(sequence), 1, 22).float()

def constructTensorFromMidiNotes(filename):
    sequence = parseNotes(filename)
    return (torch.tensor(sequence) - 1).view(len(sequence), 1, 30).float()

rnn = RNN(30, 30)

filename = 'MIDI/melody/ashover1.mid'
ashover1 = parseNotes('MIDI/melody/ashover1.mid')
tpb = getTpb('MIDI/melody/ashover1.mid')

criterion = nn.MSELoss()
learning_rate = 0.05

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

optimizer = optim.Adam(rnn.parameters(), lr = learning_rate)

start = time.time()

def train(filenames, numEpochs = 10):
    for f in filenames:
        print('Training on file: %s' % f)
        sequence = constructTensorFromMidiNotes(f)
        compSeq = sequence
        # print(sequence.size())
        tpb = getTpb(f)
        for i in range(numEpochs):
            rnn.zero_grad()
            output = torch.rand(1, 1, 30)
            for j in range(sequence.size()[0]):
                output = torch.cat((output, rnn(sequence[j]).view(1, 1, 30)))
            output = output[1:]
            loss = criterion(output, compSeq)
                # print(loss)
            loss.backward(retain_graph = True)
            optimizer.step()
            print('Epoch: %d / %d, Elapsed time: %s' % (i + 1, numEpochs, timeSince(start)))
            # mid = createMidiFileNotes('f_epoch_%d.mid' % i, output, tpb)   
    mid = createMidiFileNotes('f_final.mid',output, tpb)
    return 

#train
filenames = findFiles('MIDI/melody/ashover1*.mid')
mid = train([filename])

#test
randInp = torch.rand(1, 30)
out = rnn(randInp).view(1, 1, 30)
for i in range(100):
    out = torch.cat((out, rnn(torch.rand(1, 30)).view(1, 1, 30)))
testMid = createMidiFileNotes('test_out.mid', out, tpb)