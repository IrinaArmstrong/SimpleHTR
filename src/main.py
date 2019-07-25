from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess


class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt' # symbols of dictionary
	fnAccuracy = '../model/accuracy.txt' # to write accuracy of NN
	fnTrain = '../data/' # place to store training data
	fnInfer = '../data/test.png' # place/img to recognize text (test)
	fnCorpus = '../data/corpus.txt' # list of recognized words 


def train(model, loader):
	"train NN"
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 5 # stop training after this number of epochs without improvement
	
	# Endless cycle for training, only ends when no improvement of character 
	# error rate occured more then number of epochs, chosen for early stopping
	while True:
		# Count epochs
		epoch += 1
		print('Epoch:', epoch)

		# Train
		print('Train NN')
		# Load train set (of 25000 images = 1 epoch)
		loader.trainSet()
		
		while loader.hasNext():
			# Get current batch index and overall number of batches
			iterInfo = loader.getIteratorInfo()
			# iterator gets next batch samples addresses, then load and preprocess images
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# Validate
		charErrorRate = validate(model, loader)
		
		# If best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0 # Reset counter
			model.save() # Sa model
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
		else:
			print('Character error rate not improved')
			noImprovementSince += 1 # Increment counter

		# Stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def validate(model, loader):
	"validate NN"
	print('Validate NN')
	# Switch DataLoader to validation set
	loader.validationSet()
	numCharErr = 0 # error rate for chars recognized wrongly
	numCharTotal = 0 # counter for chats total
	numWordOK = 0 # counter for words recognized rightly
	numWordTotal = 0 # counter for words total
	while loader.hasNext():
		# Print info about training process
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		# Get new batch
		batch = loader.getNext()
		# Feed a batch into the NN to recognize the texts
		# Returns: (texts, probs)
		(recognized, _) = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			# TODO: upper/lower case - ???
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			#  edit distance - measure of distinction between two words
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	
	# Print validation result
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate

# TODO: make search in dir and recognize texts on ALL images found
def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:', '"' + recognized[0] + '"')
	print('Probability:', probability[0])


def main():
	"main function"
	# Parse optional command line args
	parser = argparse.ArgumentParser()
	# add_argument() - specify which command-line options the program is willing to accept.
	# action - function, that will be executed, when appropriate argument received
	# store_true - option, if appropriate argument received, then = true
	parser.add_argument('--train', help='train the NN', action='store_true')
	parser.add_argument('--validate', help='validate the NN', action='store_true')
	parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
	parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
	parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')

	# parse_args() method actually returns some data from the options specified
	args = parser.parse_args()

	# Go through received arguments
	decoderType = DecoderType.BestPath # bestPath -> default decoder
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch
	elif args.wordbeamsearch:
		decoderType = DecoderType.WordBeamSearch

	# Train or validate on IAM dataset	
	if args.train or args.validate:
		# Load training data, create TF model
		loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

		# Save characters of model for inference mode
		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		
		# Save words contained in dataset into file
		open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

		# Execute training or validation
		if args.train:
			model = Model(loader.charList, decoderType)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, decoderType, mustRestore=True)
			validate(model, loader)

	# Infer text on test image
	else:
		print(open(FilePaths.fnAccuracy).read())
		model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
		infer(model, FilePaths.fnInfer)


if __name__ == '__main__':
	main()

