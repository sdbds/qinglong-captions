import random

class NameSampler:

    def __init__(self, name_prob_dict, sample_num=2048) -> None:
        self.name_prob_dict = name_prob_dict
        self._id2name = list(name_prob_dict.keys())
        self.sample_ids = []

        total_prob = 0.
        for ii, (_, prob) in enumerate(name_prob_dict.items()):
            tgt_num = int(prob * sample_num)
            total_prob += prob
            if tgt_num > 0:
                self.sample_ids += [ii] * tgt_num

        nsamples = len(self.sample_ids)
        assert prob <= 1
        if prob < 1 and nsamples < sample_num:
            self.sample_ids += [len(self._id2name)] * (sample_num - nsamples)
            self._id2name.append('_')

    def sample(self) -> str:
        return self._id2name[random.choice(self.sample_ids)]