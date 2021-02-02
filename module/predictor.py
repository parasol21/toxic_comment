import csv

# parse in already trained model
class Predictor(object):
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model

    def predict(self, test_x):
        predictions = self.model.predict(test_x)
        return predictions

    def predict_prob(self, test_x):
        prob = self.model.predict_prob(test_x)
        return prob

    def save_result(self, test_ids, probs):
        with open(self.config['output_path'], 'w') as output_csv_file:
            header = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            # create a writer with file name
            writer = csv.writer(output_csv_file)
            # create header column header
            writer.writerow(header)
            for test_id, prob in zip(test_ids, probs.tolist()):
                # write row by row
                writer.writerow([test_id] + prob)