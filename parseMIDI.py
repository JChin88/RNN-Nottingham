from mido import MidiFile, MidiTrack, Message

#return the ticks per beat
def getTpb(filename):
	return MidiFile(filename).ticks_per_beat

# Parse a MIDI file by finding its notes and lengths
# Note Representation: [note, velocity, length, time]
def parseNotes(filename, tracknumber = 0, subdivisions = 2):
	sequence = []
	mid = MidiFile(filename)
	tpb = mid.ticks_per_beat
	divisions = int(tpb / subdivisions)
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
					sequence += [noteToBin(msg.note) + lenToBin(length, tpb)]
					# sequence += [[msg.note, msg.velocity, length, elapsedTime]]
					noteOff = True
				j += 1
	return sequence

# Parse a MIDI file by reading it by time(beats)
def parseBeats(filename, tracknumber = 0, subdivisions = 2):
	sequence = []
	mid = MidiFile(filename)
	track = mid.tracks[tracknumber]
	tpb = mid.ticks_per_beat
	divisions = int(tpb / subdivisions)
	curTone = 0
	numticks = 0
	for msg in track:
		numticks = 0
		while numticks < msg.time:
			sequence += [noteToBin(curTone)]
			numticks += divisions
		if msg.type == 'note_on':
			curTone = msg.note
			curVel = msg.velocity
		elif msg.type == 'note_off':
			curTone = 0
			curVel = 0
	return sequence

#converts a note into a binary representation
def noteToBin(tone):
	noteBin = []
	note = tone % 12
	pitch = int(tone/12)
	for i in range(22):
		if i == note or i == pitch + 12:
			noteBin += [1]
		else:
			noteBin += [0]
	return noteBin

#converts a binary value list of length 16 into a note
def binToNote(binaryList):	if len(binaryList) != 22:
		raise ValueError('list must have length 22')
	note = [i for i, j in enumerate(binaryList[:12]) if j == max(binaryList[:12])][0]
	pitch = [i for i, j in enumerate(binaryList[12:]) if j == max(binaryList[12:])][0]
	tone = pitch*12 + note
	return (tone, 90)

def lenToBin(length, tpb, subdivisions = 2):
	lenBin = []
	l = length / (tpb / subdivisions)
	for i in range(4 * subdivisions):
		if l == i + 1:
			lenBin += [1]
		else:
			lenBin += [0]
	return lenBin

def binToNoteLength(binaryList, tpb, subdivisions = 2):
	if len(binaryList) != 22 + 4*subdivisions:
		raise ValueError('list must be correct size')
	tone, velocity = binToNote(binaryList[:22])
	length = [i + 1 for i, j in enumerate(binaryList[22:]) if j == max(binaryList[22:])][0]
	return (tone, velocity, length)

#reconstructs a MIDI file from a list of notes
def reconstructFromNotes(filename, sequence, tpb, subdivisions = 2):
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	mid.ticks_per_beat = tpb

	for note in sequence:
		tone, velocity, length = binToNoteLength(note, tpb, subdivisions)

		track.append(Message('note_on', note = tone, velocity = velocity, time = 0))
		track.append(Message('note_off', note = tone, velocity = 0, time = length * int(tpb/subdivisions)))

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

def reconstruct(filename, sequence, tpb, subdivisions = 2):
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	mid.ticks_per_beat = tpb

	for note in sequence:
		tone, velocity, length = note

		track.append(Message('note_on', note = tone, velocity = velocity, time = 0))
		track.append(Message('note_off', note = tone, velocity = 0, time = length * int(tpb/subdivisions)))

	mid.save('output/' + filename)
	return mid	
