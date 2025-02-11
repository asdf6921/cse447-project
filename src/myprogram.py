#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv
from collections import defaultdict, Counter
import pickle
import json


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    @classmethod
    def __init__(self, n):
        self.n = n  # Size of n-grams
        self.model = defaultdict(Counter)

    @classmethod
    def load_training_data(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            data = []
            for row in reader:
                _, unicode_seq, frequency, language_id = row  # Ignore the Word column
                unicode_seq = eval(unicode_seq)
                frequency = int(frequency)
                language_id = int(language_id)
                data.append((unicode_seq, frequency, language_id))
            return data

    @classmethod
    def load_test_data(cls, file_path):
        # your code here
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            data = []
            for row in reader:
                _, unicode_seq, frequency, language_id = row  # Ignore the Word column
                unicode_seq = eval(unicode_seq)
                frequency = int(frequency)
                language_id = int(language_id)
                data.append((unicode_seq, frequency, language_id))
            return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data):
        for unicode_seq, frequency, language_id in data:
            for i in range(len(unicode_seq) - self.n + 1):
                prefix = unicode_seq[i:i + self.n - 1]
                next_char_unicode = unicode_seq[i + self.n - 1]
                self.model[prefix][next_char_unicode] += frequency

    def run_pred(self, sequence_unicode, top_k=3):
        # your code here
        sequence_unicode = sequence_unicode[-(self.n - 1):]
        if sequence_unicode in self.model:
            predictions = self.model[sequence_unicode].most_common(top_k)
            return [char for char, _ in predictions]
        return []

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')


    @classmethod
    def load(cls, work_dir):
        # your code here
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()
# Quick method to return the character from the unicode
def unicode_to_char(unicode):
    return chr(unicode)

# Small TEST - this works!
model = MyModel(n=4)
data = model.load_training_data("./src/train_split_en.csv")  # Load data
model.run_train(data)  # Train the model

# Predict next Unicode characters for the sequence "hel"
predictions = model.run_pred((104, 101, 108))
predictions = [unicode_to_char(p) for p in predictions]
print(predictions)
model.write_pred(predictions, 'output.txt')

# Will work once we update to handle larger dataset -> think we need to update run_pred
# i think we need to make sure this works tho
# if __name__ == '__main__':
#     parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
#     parser.add_argument('mode', choices=('train', 'test'), help='what to run')
#     parser.add_argument('--work_dir', help='where to save', default='work')
#     parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
#     parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
#     args = parser.parse_args()

#     random.seed(0)

#     if args.mode == 'train':
#         if not os.path.isdir(args.work_dir):
#             print('Making working directory {}'.format(args.work_dir))
#             os.makedirs(args.work_dir)

#         print('Instatiating model')
#         model = MyModel(n=4)

#         print('Loading training data')
#         train_data = MyModel.load_training_data('train_split_en.csv')

#         print('Training')
#         model.run_train(train_data)

#         print('Saving model')
#         model.save(args.work_dir)

#     elif args.mode == 'test':
#         print('Loading model')
#         model = MyModel.load('work/model.checkpoint')

#         print('Loading test data from {}'.format(args.test_data))
#         test_data = MyModel.load_test_data(args.test_data)

#         print('Making predictions')
#         pred = model.run_pred(test_data)

#         print('Writing predictions to {}'.format(args.test_output))
#         assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
#         model.write_pred(pred, args.test_output)
#     else:
#         raise NotImplementedError('Unknown mode {}'.format(args.mode))
