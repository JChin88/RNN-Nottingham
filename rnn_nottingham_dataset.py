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
        # print(out)
        o = out[0]
        note = [i for i, j in enumerate(o[:12]) if j == max(o[:12])][0]
        pitch = [i for i, j in enumerate(o[12:22]) if j == max(o[12:22])][0]
        length = [i for i, j in enumerate(o[22:]) if j == max(o[22:])][0]
        print((pitch*12 + note, 90, length))
        sequence += [(pitch*12 + note, 90, length)]
    # print(sequence)
    print('\n')
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
inp = constructTensorFromMidiNotes(filename)
# print(inp.size())


# sequence = constructTensorFromMidi(filename)
# print(output[0][0])
# print(sequence.long()[1][0])

# mid = createMidiFile('test.mid',output, tpb)

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
# output = torch.rand(1, 1, 30)
# print(rnn(inp[0]).size())
# for i in range(len(inp)):
#     output = torch.cat((output, rnn(inp[i]).view(1, 1, 30)))
# # print(inp)
# # print(output)
# output = output[1:]
# loss = criterion(output, inp)
# loss.backward()
# params = rnn.parameters()
# # print(len(list(params)))
# for p in params:
#     print(p.data)
#     p.data.add_(-learning_rate, p.grad.data)
#     print(p.data)

start = time.time()
# for j in range(sequence.size()[0] - 1):
#     output += [rnn(sequence[j])]
#     loss = criterion(output[j][0], compSeq[j + 1][0])
#     print(loss)
#     loss.backward(retain_graph = True)
#     for p in rnn.parameters():
#         p.data.add_(-learning_rate, p.grad.data)

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

filenames = findFiles('MIDI/melody/ashover1*.mid')
mid = train([filename])
# for msg in mid.tracks[0]:
#     print(msg)

#test
randInp = torch.rand(1, 30)
out = rnn(randInp).view(1, 1, 30)
for i in range(100):
    # print((out[i] +  1).int().numpy())
    out = torch.cat((out, rnn(torch.rand(1, 30)).view(1, 1, 30)))
    # out = torch.cat((out, rnn(torch.rand(1, 30)).view(1, 1, 30)))
testMid = createMidiFileNotes('test_out.mid', out, tpb)
# for msg in testMid.tracks[0]:
#     print(msg)


# rnn = nn.LSTMCell(10, 20)
# input = torch.randn(6, 3, 10)
# hx = torch.randn(3, 20)
# cx = torch.randn(3, 20)
# output = []
# for i in range(6):
#         hx, cx = rnn(input[i], (hx, cx))
#         output.append(hx)

# print(output)