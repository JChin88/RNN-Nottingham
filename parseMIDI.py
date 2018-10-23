from mido import MidiFile, MidiTrack, Message
import torch

# Parse a MIDI file by finding its notes and lengths
# Note Representation: [note, velocity, length, time]
def parseNotes(filename, tracknumber = 0):
	sequence = []
	mid = MidiFile(filename)
	track = mid.tracks[tracknumber]
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
					sequence += [[msg.note, msg.velocity, length, elapsedTime]]
					noteOff = True
				j += 1
	return torch.tensor(sequence)

# Parse a MIDI file by reading it by time(beats)
def parseBeats(filename):
	#TODO: implement parsing of beats
	# Look for Time Signature Message
	return

#reconstructs a MIDI file from a list of note tensors
def reconstructFromNotes(filename, sequence, ticks_per_beat):
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	mid.ticks_per_beat = ticks_per_beat

	elapsedTime = 0
	noteOffs = []

	for note in sequence:
		tone = note[0].item()
		velocity = note[1].item()
		length = note[2].item()
		time = note[3].item() - elapsedTime

		for noteOff in noteOffs:
			if noteOff[1] <= time:
				elapsedTime += noteOff[1]
				time -= noteOff[1]
				track.append(noteOff[0])
				noteOffs.remove(noteOff)

		track.append(Message('note_on', note = tone, velocity = velocity, time = time))
		elapsedTime += time
		noteOffs += [(Message('note_off', note = tone, velocity = 0, time = length), length)]

	mid.save('output/' + filename)
	return mid

filename = 'MIDI/melody/ashover1.mid'
mid = MidiFile(filename)
print('\nOriginal')
track1 = mid.tracks[0]
for msg in track1:
	print(msg)
	
tpb = mid.ticks_per_beat

mid.save('output/realashover1.mid')

metaMessages = track1[0:4]

sequence = parseNotes(filename)
# print(sequence.size())
# for note in sequence:
# 	print(note)

print('\nReconstructed:')
mid = reconstructFromNotes('testashover1.mid', sequence, tpb)
track2 = mid.tracks[0]
for msg in track2:
	print(msg)


# for i in range(len(track2)):
# 	if track1[i] != track2[i]:
# 		print(i)
# 		print('\nTrack 1: ')
# 		print(track1[i])
# 		print('\nTrack 2: ')
# 		print(track2[i])
# 		print('\n')