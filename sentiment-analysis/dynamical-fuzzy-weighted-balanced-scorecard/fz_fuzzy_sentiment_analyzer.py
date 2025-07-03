import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzySentimentAnalyzer:
    def __init__(self, positive_keywords, negative_keywords):
        self.positive_keywords = set(positive_keywords)
        self.negative_keywords = set(negative_keywords)
        self._build_fuzzy_system()

    def _count_words(self, text):
        words = text.split()
        pos_count = sum(1 for word in words if word in self.positive_keywords)
        neg_count = sum(1 for word in words if word in self.negative_keywords)
        return pos_count, neg_count

    def _build_fuzzy_system(self):
        self.kata_positif = ctrl.Antecedent(np.arange(0, 11, 1), 'kata_positif')
        self.kata_negatif = ctrl.Antecedent(np.arange(0, 11, 1), 'kata_negatif')
        self.rating_bintang = ctrl.Antecedent(np.arange(1, 6, 1), 'rating_bintang')
        self.skor_sentimen = ctrl.Consequent(np.arange(0, 101, 1), 'skor_sentimen')
        
        self.kata_positif.automf(names=['sedikit', 'sedang', 'banyak'])
        self.kata_negatif.automf(names=['sedikit', 'sedang', 'banyak'])
        self.rating_bintang['buruk'] = fuzz.trimf(self.rating_bintang.universe, [1, 1, 3])
        self.rating_bintang['cukup'] = fuzz.trimf(self.rating_bintang.universe, [2, 3, 4])
        self.rating_bintang['baik'] = fuzz.trimf(self.rating_bintang.universe, [3, 5, 5])
        self.skor_sentimen.automf(names=['sangat_buruk', 'buruk', 'sedang', 'baik', 'sangat_baik'])
        
        # 27 rules berdasarkan kombinasi rating bintang, kata positif, dan kata negatif
        rule_list = [
            ctrl.Rule(self.rating_bintang['baik'] & self.kata_positif['banyak'] & self.kata_negatif['sedikit'], self.skor_sentimen['sangat_baik']),
            ctrl.Rule(self.rating_bintang['baik'] & self.kata_positif['banyak'] & self.kata_negatif['sedang'], self.skor_sentimen['baik']),
            ctrl.Rule(self.rating_bintang['baik'] & self.kata_positif['banyak'] & self.kata_negatif['banyak'], self.skor_sentimen['sedang']),
            ctrl.Rule(self.rating_bintang['baik'] & self.kata_positif['sedang'] & self.kata_negatif['sedikit'], self.skor_sentimen['baik']),
            ctrl.Rule(self.rating_bintang['baik'] & self.kata_positif['sedang'] & self.kata_negatif['sedang'], self.skor_sentimen['baik']),
            ctrl.Rule(self.rating_bintang['baik'] & self.kata_positif['sedang'] & self.kata_negatif['banyak'], self.skor_sentimen['sedang']),
            ctrl.Rule(self.rating_bintang['baik'] & self.kata_positif['sedikit'] & self.kata_negatif['sedikit'], self.skor_sentimen['baik']),
            ctrl.Rule(self.rating_bintang['baik'] & self.kata_positif['sedikit'] & self.kata_negatif['sedang'], self.skor_sentimen['sedang']),
            ctrl.Rule(self.rating_bintang['baik'] & self.kata_positif['sedikit'] & self.kata_negatif['banyak'], self.skor_sentimen['buruk']),
            ctrl.Rule(self.rating_bintang['cukup'] & self.kata_positif['banyak'] & self.kata_negatif['sedikit'], self.skor_sentimen['baik']),
            ctrl.Rule(self.rating_bintang['cukup'] & self.kata_positif['banyak'] & self.kata_negatif['sedang'], self.skor_sentimen['sedang']),
            ctrl.Rule(self.rating_bintang['cukup'] & self.kata_positif['banyak'] & self.kata_negatif['banyak'], self.skor_sentimen['buruk']),
            ctrl.Rule(self.rating_bintang['cukup'] & self.kata_positif['sedang'] & self.kata_negatif['sedikit'], self.skor_sentimen['sedang']),
            ctrl.Rule(self.rating_bintang['cukup'] & self.kata_positif['sedang'] & self.kata_negatif['sedang'], self.skor_sentimen['sedang']),
            ctrl.Rule(self.rating_bintang['cukup'] & self.kata_positif['sedang'] & self.kata_negatif['banyak'], self.skor_sentimen['buruk']),
            ctrl.Rule(self.rating_bintang['cukup'] & self.kata_positif['sedikit'] & self.kata_negatif['sedikit'], self.skor_sentimen['sedang']),
            ctrl.Rule(self.rating_bintang['cukup'] & self.kata_positif['sedikit'] & self.kata_negatif['sedang'], self.skor_sentimen['buruk']),
            ctrl.Rule(self.rating_bintang['cukup'] & self.kata_positif['sedikit'] & self.kata_negatif['banyak'], self.skor_sentimen['sangat_buruk']),
            ctrl.Rule(self.rating_bintang['buruk'] & self.kata_positif['banyak'] & self.kata_negatif['sedikit'], self.skor_sentimen['sedang']),
            ctrl.Rule(self.rating_bintang['buruk'] & self.kata_positif['banyak'] & self.kata_negatif['sedang'], self.skor_sentimen['buruk']),
            ctrl.Rule(self.rating_bintang['buruk'] & self.kata_positif['banyak'] & self.kata_negatif['banyak'], self.skor_sentimen['buruk']),
            ctrl.Rule(self.rating_bintang['buruk'] & self.kata_positif['sedang'] & self.kata_negatif['sedikit'], self.skor_sentimen['buruk']),
            ctrl.Rule(self.rating_bintang['buruk'] & self.kata_positif['sedang'] & self.kata_negatif['sedang'], self.skor_sentimen['sangat_buruk']),
            ctrl.Rule(self.rating_bintang['buruk'] & self.kata_positif['sedang'] & self.kata_negatif['banyak'], self.skor_sentimen['sangat_buruk']),
            ctrl.Rule(self.rating_bintang['buruk'] & self.kata_positif['sedikit'] & self.kata_negatif['sedikit'], self.skor_sentimen['buruk']),
            ctrl.Rule(self.rating_bintang['buruk'] & self.kata_positif['sedikit'] & self.kata_negatif['sedang'], self.skor_sentimen['sangat_buruk']),
            ctrl.Rule(self.rating_bintang['buruk'] & self.kata_positif['sedikit'] & self.kata_negatif['banyak'], self.skor_sentimen['sangat_buruk'])
        ]
        self.sentiment_ctrl = ctrl.ControlSystem(rule_list)
        self.sentiment_simulation = ctrl.ControlSystemSimulation(self.sentiment_ctrl)

    def analyze(self, text, rating):
        pos_count, neg_count = self._count_words(text)
        self.sentiment_simulation.input['kata_positif'] = pos_count
        self.sentiment_simulation.input['kata_negatif'] = neg_count
        self.sentiment_simulation.input['rating_bintang'] = rating
        self.sentiment_simulation.compute()
        return self.sentiment_simulation.output['skor_sentimen']