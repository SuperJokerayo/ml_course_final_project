import yaml

class Base_Model(object):
    def __init__(self, train_data, valid_data, test_data, hyperparams_path):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.hyperparams_path = hyperparams_path
        self.hyperparams = self._read_hyperparams()

    def _generate_feature_and_target(self):
        raise NotImplementedError

    def _read_hyperparams(self):
        with open(self.hyperparams_path, 'r', encoding = "utf-8") as f:
            return yaml.load(f, Loader = yaml.FullLoader)

    def train_and_eval(self):
        raise NotImplementedError

    def test(self, model_path = None):
        raise NotImplementedError

    def save_model(self, model_path):
        raise NotImplementedError

    def load_model(self, model_path):
        raise NotImplementedError