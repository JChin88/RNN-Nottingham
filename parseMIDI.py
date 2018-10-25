from mido import MidiFile, MidiTrack, Message

#return the ticks per beat
def getTpb(filename):
	return MidiFile(filename).ticks_per_beat

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
	return torch.tensor(sequence).float()

# Parse a MIDI file by reading it by time(beats)
def parseBeats(filename, tracknumber = 0, subdivisions = 2):
	#TODO: implement parsing of beats
	# Look for ticks per beat
	sequence = []
	mid = MidiFile(filename)
	track = mid.tracks[tracknumber]
	tpb = mid.ticks_per_beat
	divisions = int(tpb / subdivisions)
	curTone = 0
	curVel = 0
	numticks = 0
	for msg in track:
		numticks = 0
		while numticks < msg.time:
			sequence += [noteToBin(curTone, curVel)]
			numticks += divisions
		if msg.type == 'note_on':
			curTone = msg.note
			curVel = msg.velocity
		elif msg.type == 'note_off':
			curTone = 0
			curVel = 0
	return sequence

#converts a note ([tone, velocity] pair) into a binary representation
def noteToBin(tone, velocity):
	btone = '{0:07b}'.format(tone)
	bvel = '{0:07b}'.format(velocity)
	note = []
	for b in btone + bvel:
		note += [int(b)]
	return note

#converts a binary value list of length 16 into a note
def binToNote(binaryList):
	# print(binaryList)
	if len(binaryList) != 14:
		raise ValueError('list must have length 14')
	note = ''.join(str(n) for n in binaryList[0:7])
	velocity = ''.join(str(v) for v in binaryList[7:])
	return (int(note, 2), int(velocity, 2))

#reconstructs a MIDI file from a list of notes
def reconstructFromNotes(filename, sequence, ticks_per_beat):
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	mid.ticks_per_beat = ticks_per_beat

	elapsedTime = 0
	noteOffs = []

	for note in sequence:
		tone = int(note[0].item())
		velocity = int(note[1].item())
		length = int(note[2].item())
		time = int(note[3].item()) - elapsedTime

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

#reconstructs a MIDI file from a list of beats
def reconstructFromBeats(filename, sequence, ticks_per_beat, subdivisions = 2):
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	mid.ticks_per_beat = ticks_per_beat
	divisions = int(ticks_per_beat / subdivisions)

	numticks = 0
	curTone = 0
	curVel = 0

	for beat in sequence:
		tone, velocity = binToNote(beat)
		# print(tone, velocity)
		if tone != curTone:
			if curTone != 0:
				track.append(Message('note_off', note = curTone, velocity = 0, time = numticks))
				numticks = 0
			track.append(Message('note_on', note = tone, velocity = velocity, time = numticks))
			numticks = 0
			curTone = tone
		numticks += divisions

	mid.save('output/' + filename)
	return mid

# # filename = '../Parker,_Charlie_-_Donna_Lee.midi'
# filename = 'MIDI/melody/ashover1.mid'
# mid = MidiFile(filename)
# print('\nOriginal')
# track1 = mid.tracks[0]
# for msg in track1:
# 	print(msg)

# tpb = mid.ticks_per_beat

# # mid.save('output/realDonnaLee.mid')
# mid.save('output/realAshover1.mid')

# sequence = parseBeats(filename)
# # for s in sequence:
# # 	print(s)

# print('\nReconstructed:')
# mid = reconstructFromBeats('testAshover1.mid', sequence, tpb)
# track2 = mid.tracks[0]
# for msg in track2:
# 	print(msg)
