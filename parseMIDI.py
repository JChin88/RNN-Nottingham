from mido import MidiFile

filename = 'NottinghamDataset/nottingham-dataset-master/MIDI/ashover1.mid'
mid = MidiFile(filename)

for i, track in enumerate(mid.tracks):
	print('Track {}: {}'.format(i, track.name))
	for msg in track:
		print(msg)