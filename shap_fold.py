from utils import load_data, split_data, split_xy, split_X_by_Y, flatten_rules, decode_rules, get_scores
from algo import shap_fold, classify, predict
import pickle


class Classifier:
    def __init__(self, attrs=None):
        self.attrs = attrs
        self.rules = None
        self.asp_rules = None
        self.seq = 0
        self.translation = None
        s = attrs[-1].split('#')
        self.label = s[0]
        self.pos = s[2]

    def fit(self, X_pos, SHAP_pos, X_neg, SHAP_neg):
        self.rules = shap_fold(X_pos, SHAP_pos, X_neg, SHAP_neg, [])

    def classify(self, x):
        return classify(self.rules, x)

    def predict(self, X):
        return predict(self.rules, X)

    def asp(self):
        if self.asp_rules is None and self.rules is not None and self.attrs is not None:
            frs = flatten_rules(self.rules)
            self.asp_rules = decode_rules(frs, self.attrs)
        return self.asp_rules

    def print_asp(self):
        for r in self.asp():
            print(r)

    def save_model_to_file(self, file):
        f = open(file, 'wb')
        pickle.dump(self, f)
        f.close()


def load_model_from_file(file):
    f = open(file, 'rb')
    ret = pickle.load(f)
    f.close()
    return ret
