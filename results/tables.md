## Part I: language models

| model | test PPL | test CE | val PPL | params / counts (M) | non-emb params (M) | train time (s) | median s/epoch | best epoch | test scoring (k tok/s) | gen ms/token |
|---|---|---|---|---|---|---|---|---|---|---|
| KN 4-gram | 216.5 | 5.378 | 229.3 | 5.77† | – | 7 (CPU) | – | – | 450 (CPU) | 1.16 |
| RNN | 153.4 | 5.033 | 165.6 | 24.45 | 0.94 | 423 | 21.2 | 20 | 241 | 1.85 |
| LSTM | 135.0 | 4.905 | 143.9 | 27.27 | 3.77 | 487 | 23.2 | 19 | 152 | 1.72 |
| Transformer | 131.0 | 4.876 | 140.4 | 21.75 | 4.37 | 584 | 29.2 | 20 | 201 | 1.39 |

† n-gram: number of stored (context, word) count entries, not trainable parameters.


## Generation repetition metrics (40 tokens; greedy: 5 prompts, top-k: 5x10 samples)

| model | decoding | distinct-1 | distinct-2 | repeated 3-gram rate |
|---|---|---|---|---|
| ngram | greedy | 0.315 | 0.415 | 0.553 |
| ngram | sample | 0.725 | 0.935 | 0.024 |
| rnn | greedy | 0.395 | 0.549 | 0.337 |
| rnn | sample | 0.696 | 0.925 | 0.027 |
| lstm | greedy | 0.355 | 0.503 | 0.426 |
| lstm | sample | 0.685 | 0.919 | 0.027 |
| transformer | greedy | 0.355 | 0.492 | 0.411 |
| transformer | sample | 0.672 | 0.903 | 0.032 |
| lstm_w2v | greedy | 0.300 | 0.410 | 0.489 |
| lstm_w2v | sample | 0.662 | 0.908 | 0.036 |
| lstm_glove | greedy | 0.315 | 0.456 | 0.474 |
| lstm_glove | sample | 0.647 | 0.897 | 0.032 |
| lstm_smallinit | greedy | 0.230 | 0.313 | 0.621 |
| lstm_smallinit | sample | 0.700 | 0.924 | 0.027 |

## Part II: LSTM embedding ablation

| embedding | test PPL | val PPL | best epoch | val PPL @ep1 | val PPL @ep5 | epochs to within 0.05 of best | final gap (val − logged train) | step-loss std | grad-norm mean/max | trainable (M) | LM train time (s) | median s/epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random N(0,1) (trainable) | 135.0 | 143.9 | 19 | 412 | 182 | 10 | 0.45 | 0.127 | 0.25/1.79 | 27.27 | 487 | 23.2 |
| random N(0,0.1²) (trainable) | 140.3 | 149.4 | 20 | 706 | 211 | 11 | 0.51 | 0.123 | 0.26/2.02 | 27.27 | 427 | 21.5 |
| own Skip-gram (frozen) | 130.4 | 138.1 | 19 | 559 | 185 | 11 | 0.40 | 0.123 | 0.26/1.83 | 18.60 | 609 | 31.2 |
| GloVe 6B (frozen) | 133.9 | 142.3 | 19 | 564 | 188 | 11 | 0.33 | 0.122 | 0.26/1.81 | 18.60 | 624 | 31.6 |

Own Skip-gram training cost (separate from LM training): 82.3s on CPU (CPU, 1 worker thread(s)), 10 epochs over 2,075,677 tokens. GloVe type coverage 96.20%, train-token coverage 95.87%, missing types 1098.


### Seed variation (test PPL over seeds 42 + extra seeds)

| model | n seeds | mean | std | values |
|---|---|---|---|---|
| lstm_glove | 3 | 132.7 | 1.1 | 133.9, 131.9, 132.3 |
| lstm | 3 | 135.0 | 0.7 | 135.0, 135.8, 134.4 |
| lstm_w2v | 3 | 130.2 | 1.9 | 130.4, 132.0, 128.3 |
| rnn | 3 | 153.2 | 0.2 | 153.4, 153.0, 153.3 |
| transformer | 3 | 131.8 | 0.6 | 131.0, 132.2, 132.0 |

## Part III: AG News (frozen LM features + logistic regression)

| LM | LM test PPL | last acc / F1 | mean acc / F1 | emb acc / F1 |
|---|---|---|---|---|
| rnn | 153.4 | 0.617 / 0.616 | 0.705 / 0.705 | 0.755 / 0.754 |
| lstm | 135.0 | 0.638 / 0.635 | 0.701 / 0.700 | 0.733 / 0.733 |
| transformer | 131.0 | 0.719 / 0.718 | 0.767 / 0.766 | 0.770 / 0.769 |
| lstm_smallinit | 140.3 | 0.644 / 0.642 | 0.731 / 0.730 | 0.765 / 0.765 |
| lstm_w2v | 130.4 | 0.776 / 0.776 | 0.803 / 0.803 | 0.824 / 0.823 |
| lstm_glove | 133.9 | 0.790 / 0.790 | 0.846 / 0.846 | 0.878 / 0.878 |

Spearman(PPL, acc): {'last': {'spearman_ppl_vs_acc': -0.7714285714285715, 'p_value': 0.07239650145772594, 'n_models': 6}, 'mean': {'spearman_ppl_vs_acc': -0.6571428571428573, 'p_value': 0.1561749271137024, 'n_models': 6}, 'emb': {'spearman_ppl_vs_acc': -0.6571428571428573, 'p_value': 0.1561749271137024, 'n_models': 6}}


OOV: {"agnews_train_subset": {"tokens": 875685, "token_oov_rate": 0.07026956040128585, "type_oov_rate": 0.5163863412302453, "mean_length": 43.78425}, "agnews_test": {"tokens": 330199, "token_oov_rate": 0.07066041992858851, "type_oov_rate": 0.41763077900276757, "mean_length": 43.44723684210526}, "train_size": 20000, "test_size": 7600, "train_label_counts": [5000, 5000, 5000, 5000], "sst2_validation": {"tokens": 17780, "token_oov_rate": 0.0703037120359955, "type_oov_rate": 0.2503493246390312, "mean_length": 20.389908256880734}}


Accuracy by per-example OOV-rate bucket: {"rnn/last": {"0": 0.5728, "(0,5%]": 0.6287, "(5,10%]": 0.6173, ">10%": 0.6233}, "rnn/mean": {"0": 0.685, "(0,5%]": 0.7238, "(5,10%]": 0.6992, ">10%": 0.7027}, "rnn/emb": {"0": 0.7327, "(0,5%]": 0.7582, "(5,10%]": 0.7543, ">10%": 0.7615}, "lstm/last": {"0": 0.6014, "(0,5%]": 0.6499, "(5,10%]": 0.6418, ">10%": 0.6346}, "lstm/mean": {"0": 0.6599, "(0,5%]": 0.7125, "(5,10%]": 0.6958, ">10%": 0.7139}, "lstm/emb": {"0": 0.7136, "(0,5%]": 0.7441, "(5,10%]": 0.726, ">10%": 0.7405}, "transformer/last": {"0": 0.679, "(0,5%]": 0.7191, "(5,10%]": 0.7167, ">10%": 0.7385}, "transformer/mean": {"0": 0.7411, "(0,5%]": 0.7743, "(5,10%]": 0.7591, ">10%": 0.7794}, "transformer/emb": {"0": 0.7399, "(0,5%]": 0.7823, "(5,10%]": 0.7591, ">10%": 0.7845}, "lstm_w2v/last": {"0": 0.7518, "(0,5%]": 0.7795, "(5,10%]": 0.7762, ">10%": 0.782}, "lstm_w2v/mean": {"0": 0.7876, "(0,5%]": 0.8167, "(5,10%]": 0.8004, ">10%": 0.7999}, "lstm_w2v/emb": {"0": 0.7995, "(0,5%]": 0.8346, "(5,10%]": 0.8235, ">10%": 0.8224}, "lstm_glove/last": {"0": 0.7697, "(0,5%]": 0.7974, "(5,10%]": 0.7975, ">10%": 0.7815}, "lstm_glove/mean": {"0": 0.8282, "(0,5%]": 0.8497, "(5,10%]": 0.8462, ">10%": 0.8495}, "lstm_glove/emb": {"0": 0.8723, "(0,5%]": 0.8864, "(5,10%]": 0.8764, ">10%": 0.8746}, "lstm_smallinit/last": {"0": 0.611, "(0,5%]": 0.6517, "(5,10%]": 0.6485, ">10%": 0.6443}, "lstm_smallinit/mean": {"0": 0.6897, "(0,5%]": 0.7479, "(5,10%]": 0.7323, ">10%": 0.7298}, "lstm_smallinit/emb": {"0": 0.753, "(0,5%]": 0.7804, "(5,10%]": 0.7606, ">10%": 0.7605}, "count": {"0": 838, "(0,5%]": 2122, "(5,10%]": 2686, ">10%": 1954}}


### Part III over LM seeds (mean / std of accuracy over checkpoints)

| family | emb mean | last mean | mean mean | emb std | last std | mean std | emb count | last count | mean count |
|---|---|---|---|---|---|---|---|---|---|
| lstm | 0.7403 | 0.6497 | 0.7098 | 0.0064 | 0.0104 | 0.0076 | 3 | 3 | 3 |
| lstm_glove | 0.8783 | 0.7943 | 0.8436 | 0.0 | 0.0039 | 0.0022 | 3 | 3 | 3 |
| lstm_smallinit | 0.7653 | 0.6442 | 0.7313 | nan | nan | nan | 1 | 1 | 1 |
| lstm_w2v | 0.8237 | 0.7666 | 0.802 | 0.0 | 0.0088 | 0.0035 | 3 | 3 | 3 |
| rnn | 0.7594 | 0.6164 | 0.7125 | 0.0041 | 0.0007 | 0.0063 | 3 | 3 | 3 |
| transformer | 0.7717 | 0.7302 | 0.7717 | 0.0037 | 0.0105 | 0.0069 | 3 | 3 | 3 |
| untrained_lstm | 0.6832 | 0.4747 | 0.7183 | nan | nan | nan | 1 | 1 | 1 |
| untrained_transformer | 0.6832 | 0.4964 | 0.65 | nan | nan | nan | 1 | 1 | 1 |

Spearman(PPL, acc) over all trained checkpoints: {'emb': {'rho': -0.636105810569984, 'p_value': 0.00807549689683057, 'n': 16}, 'last': {'rho': -0.7588235294117647, 'p_value': 0.0006540302361647908, 'n': 16}, 'mean': {'rho': -0.6588235294117648, 'p_value': 0.005511422336343683, 'n': 16}}


Spearman within Part I families: {'emb': {'rho': -0.45, 'p_value': 0.22421610749233675, 'n': 9}, 'last': {'rho': -0.8166666666666667, 'p_value': 0.0072247852463587885, 'n': 9}, 'mean': {'rho': -0.5666666666666667, 'p_value': 0.1116329876114912, 'n': 9}}


TF-IDF bag-of-words reference: {'C': 1.0, 'test_accuracy': 0.8852631578947369, 'test_macro_f1': 0.8849570556443742}

