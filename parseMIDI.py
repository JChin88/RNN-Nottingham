from mido import MidiFile
import torch

# Parse a MIDI file by finding its notes and lengths
# Note Representation: [note, velocity, length, time]
def parseNotes(filename):
	sequence = []
	mid = MidiFile(filename)
	track = mid.tracks[0]
	elapsedTime = 0
	for i, msg in enumerate(track):
		elapsedTime += msg.time
		if msg.type == 'note_on':
			j = i + 1
			length = 0
			noteOff = False
			while not noteOff and j < len(track):
				nxtmsg = track[j]
				length += nxtmsg.time
				if nxtmsg.type == 'note_off' and nxtmsg.note == msg.note:
					sequence += [noteToTensor([msg.note, msg.velocity, length, elapsedTime])]
					noteOff = True
				j += 1
	return sequence

# Parse a MIDI file by reading it by time(beats)
def parseBeats(filename):
	#TODO: implement parsing of beats
	# Look for Time Signature Message
	return

def noteToTensor(note):
	return torch.tensor(note)

filename = 'MIDI/ashover1.mid'
mid = MidiFile(filename)

sequence = parseNotes(filename)
for note in sequence:
	print(note)
