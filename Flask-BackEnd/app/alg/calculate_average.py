import re

# Data string provided (assuming it continues with more epochs)
data_str = """
epoch 1 | training: loss: 0.6541, acc: 0.5662, auc:  0.5346 | testing: loss: 0.6262, acc: 0.6442, auc:  0.5940 | time: 72.28s, average batch time: 18.07s
epoch 2 | training: loss: 0.6328, acc: 0.6241, auc:  0.5875 | testing: loss: 0.6331, acc: 0.6146, auc:  0.5711 | time: 69.64s, average batch time: 17.41s
epoch 3 | training: loss: 0.6267, acc: 0.6529, auc:  0.6108 | testing: loss: 0.6052, acc: 0.6429, auc:  0.5855 | time: 67.94s, average batch time: 16.99s
epoch 4 | training: loss: 0.6190, acc: 0.6537, auc:  0.6169 | testing: loss: 0.6007, acc: 0.6894, auc:  0.6557 | time: 67.43s, average batch time: 16.86s
epoch 5 | training: loss: 0.6191, acc: 0.6651, auc:  0.6233 | testing: loss: 0.5998, acc: 0.6738, auc:  0.6406 | time: 68.35s, average batch time: 17.09s
epoch 0 - 5 | training: loss: 0.6303, acc: 0.6324, auc:  0.5946 | testing: loss: 0.6130, acc: 0.6530, auc:  0.6094
epoch 6 | training: loss: 0.6103, acc: 0.6723, auc:  0.6321 | testing: loss: 0.6028, acc: 0.6744, auc:  0.6323 | time: 66.87s, average batch time: 16.72s
epoch 7 | training: loss: 0.6040, acc: 0.6847, auc:  0.6422 | testing: loss: 0.5954, acc: 0.6824, auc:  0.6448 | time: 67.16s, average batch time: 16.79s
epoch 8 | training: loss: 0.6040, acc: 0.6957, auc:  0.6518 | testing: loss: 0.5925, acc: 0.6854, auc:  0.6555 | time: 67.18s, average batch time: 16.79s
epoch 9 | training: loss: 0.5983, acc: 0.6910, auc:  0.6515 | testing: loss: 0.6108, acc: 0.6956, auc:  0.6518 | time: 67.18s, average batch time: 16.80s
epoch 10 | training: loss: 0.6094, acc: 0.6991, auc:  0.6520 | testing: loss: 0.5827, acc: 0.7167, auc:  0.6549 | time: 66.99s, average batch time: 16.75s
epoch 5 - 10 | training: loss: 0.6052, acc: 0.6886, auc:  0.6459 | testing: loss: 0.5968, acc: 0.6909, auc:  0.6479
epoch 11 | training: loss: 0.5973, acc: 0.7098, auc:  0.6601 | testing: loss: 0.5970, acc: 0.7117, auc:  0.6736 | time: 66.86s, average batch time: 16.72s
epoch 12 | training: loss: 0.6041, acc: 0.7081, auc:  0.6665 | testing: loss: 0.5764, acc: 0.7066, auc:  0.6614 | time: 66.69s, average batch time: 16.67s
epoch 13 | training: loss: 0.5947, acc: 0.7150, auc:  0.6692 | testing: loss: 0.5971, acc: 0.7121, auc:  0.6611 | time: 67.14s, average batch time: 16.78s
epoch 14 | training: loss: 0.5910, acc: 0.7242, auc:  0.6716 | testing: loss: 0.5984, acc: 0.6976, auc:  0.6696 | time: 67.04s, average batch time: 16.76s
epoch 15 | training: loss: 0.5975, acc: 0.7162, auc:  0.6750 | testing: loss: 0.5571, acc: 0.7679, auc:  0.6965 | time: 67.03s, average batch time: 16.76s
epoch 10 - 15 | training: loss: 0.5969, acc: 0.7147, auc:  0.6685 | testing: loss: 0.5852, acc: 0.7192, auc:  0.6724
epoch 16 | training: loss: 0.5880, acc: 0.7226, auc:  0.6816 | testing: loss: 0.5778, acc: 0.7354, auc:  0.6911 | time: 67.48s, average batch time: 16.87s
epoch 17 | training: loss: 0.5934, acc: 0.7254, auc:  0.6773 | testing: loss: 0.5862, acc: 0.7077, auc:  0.6690 | time: 66.28s, average batch time: 16.57s
epoch 18 | training: loss: 0.5885, acc: 0.7182, auc:  0.6821 | testing: loss: 0.5875, acc: 0.7240, auc:  0.6875 | time: 66.34s, average batch time: 16.59s
epoch 19 | training: loss: 0.5943, acc: 0.7246, auc:  0.6872 | testing: loss: 0.5647, acc: 0.7379, auc:  0.6857 | time: 66.97s, average batch time: 16.74s
epoch 20 | training: loss: 0.5850, acc: 0.7201, auc:  0.6809 | testing: loss: 0.5759, acc: 0.7417, auc:  0.7057 | time: 66.86s, average batch time: 16.72s
epoch 15 - 20 | training: loss: 0.5898, acc: 0.7222, auc:  0.6818 | testing: loss: 0.5784, acc: 0.7293, auc:  0.6878
epoch 21 | training: loss: 0.5883, acc: 0.7309, auc:  0.6898 | testing: loss: 0.5734, acc: 0.7288, auc:  0.6878 | time: 67.17s, average batch time: 16.79s
epoch 22 | training: loss: 0.5781, acc: 0.7372, auc:  0.6894 | testing: loss: 0.5787, acc: 0.7517, auc:  0.7177 | time: 66.59s, average batch time: 16.65s
epoch 23 | training: loss: 0.5824, acc: 0.7379, auc:  0.6966 | testing: loss: 0.5800, acc: 0.7465, auc:  0.7040 | time: 66.92s, average batch time: 16.73s
epoch 24 | training: loss: 0.5856, acc: 0.7336, auc:  0.7002 | testing: loss: 0.5767, acc: 0.7481, auc:  0.7028 | time: 66.93s, average batch time: 16.73s
epoch 25 | training: loss: 0.5826, acc: 0.7427, auc:  0.7102 | testing: loss: 0.5695, acc: 0.7310, auc:  0.6610 | time: 67.40s, average batch time: 16.85s
epoch 20 - 25 | training: loss: 0.5834, acc: 0.7365, auc:  0.6972 | testing: loss: 0.5756, acc: 0.7412, auc:  0.6947
epoch 26 | training: loss: 0.5824, acc: 0.7462, auc:  0.7024 | testing: loss: 0.5628, acc: 0.7431, auc:  0.6864 | time: 66.79s, average batch time: 16.70s
epoch 27 | training: loss: 0.5780, acc: 0.7473, auc:  0.7047 | testing: loss: 0.5901, acc: 0.7368, auc:  0.6915 | time: 67.12s, average batch time: 16.78s
epoch 28 | training: loss: 0.5812, acc: 0.7475, auc:  0.7037 | testing: loss: 0.5576, acc: 0.7435, auc:  0.6982 | time: 66.86s, average batch time: 16.71s
epoch 29 | training: loss: 0.5791, acc: 0.7423, auc:  0.7032 | testing: loss: 0.5616, acc: 0.7516, auc:  0.7113 | time: 67.76s, average batch time: 16.94s
epoch 30 | training: loss: 0.5833, acc: 0.7344, auc:  0.6928 | testing: loss: 0.5723, acc: 0.7520, auc:  0.7114 | time: 66.66s, average batch time: 16.67s
epoch 25 - 30 | training: loss: 0.5808, acc: 0.7436, auc:  0.7014 | testing: loss: 0.5689, acc: 0.7454, auc:  0.6998
epoch 31 | training: loss: 0.5833, acc: 0.7388, auc:  0.7092 | testing: loss: 0.5719, acc: 0.7428, auc:  0.6972 | time: 66.87s, average batch time: 16.72s
epoch 32 | training: loss: 0.5768, acc: 0.7515, auc:  0.7123 | testing: loss: 0.5684, acc: 0.7422, auc:  0.7153 | time: 67.24s, average batch time: 16.81s
epoch 33 | training: loss: 0.5821, acc: 0.7410, auc:  0.7145 | testing: loss: 0.5693, acc: 0.7440, auc:  0.6997 | time: 66.70s, average batch time: 16.67s
epoch 34 | training: loss: 0.5757, acc: 0.7481, auc:  0.7057 | testing: loss: 0.5849, acc: 0.7348, auc:  0.7102 | time: 67.47s, average batch time: 16.87s
epoch 35 | training: loss: 0.5748, acc: 0.7498, auc:  0.7079 | testing: loss: 0.5548, acc: 0.7683, auc:  0.7241 | time: 66.92s, average batch time: 16.73s
epoch 30 - 35 | training: loss: 0.5786, acc: 0.7458, auc:  0.7099 | testing: loss: 0.5699, acc: 0.7464, auc:  0.7093
epoch 36 | training: loss: 0.5756, acc: 0.7533, auc:  0.7195 | testing: loss: 0.5584, acc: 0.7583, auc:  0.7084 | time: 68.07s, average batch time: 17.02s
epoch 37 | training: loss: 0.5770, acc: 0.7566, auc:  0.7260 | testing: loss: 0.5713, acc: 0.7557, auc:  0.7104 | time: 67.07s, average batch time: 16.77s
epoch 38 | training: loss: 0.5778, acc: 0.7546, auc:  0.7204 | testing: loss: 0.5489, acc: 0.7692, auc:  0.7157 | time: 66.86s, average batch time: 16.72s
epoch 39 | training: loss: 0.5702, acc: 0.7601, auc:  0.7211 | testing: loss: 0.5859, acc: 0.7477, auc:  0.7064 | time: 66.51s, average batch time: 16.63s
epoch 40 | training: loss: 0.5732, acc: 0.7554, auc:  0.7096 | testing: loss: 0.5716, acc: 0.7620, auc:  0.7310 | time: 67.49s, average batch time: 16.87s
epoch 35 - 40 | training: loss: 0.5748, acc: 0.7560, auc:  0.7193 | testing: loss: 0.5672, acc: 0.7586, auc:  0.7144
epoch 41 | training: loss: 0.5731, acc: 0.7556, auc:  0.7143 | testing: loss: 0.5663, acc: 0.7594, auc:  0.7211 | time: 67.48s, average batch time: 16.87s
epoch 42 | training: loss: 0.5753, acc: 0.7571, auc:  0.7226 | testing: loss: 0.5529, acc: 0.7657, auc:  0.7192 | time: 68.14s, average batch time: 17.04s
epoch 43 | training: loss: 0.5699, acc: 0.7614, auc:  0.7220 | testing: loss: 0.5709, acc: 0.7629, auc:  0.7346 | time: 67.50s, average batch time: 16.87s
epoch 44 | training: loss: 0.5737, acc: 0.7536, auc:  0.7233 | testing: loss: 0.5577, acc: 0.7687, auc:  0.7202 | time: 67.27s, average batch time: 16.82s
epoch 45 | training: loss: 0.5707, acc: 0.7580, auc:  0.7240 | testing: loss: 0.5660, acc: 0.7307, auc:  0.6895 | time: 67.26s, average batch time: 16.82s
epoch 40 - 45 | training: loss: 0.5725, acc: 0.7571, auc:  0.7212 | testing: loss: 0.5628, acc: 0.7575, auc:  0.7169
epoch 46 | training: loss: 0.5734, acc: 0.7531, auc:  0.7280 | testing: loss: 0.5486, acc: 0.7524, auc:  0.6939 | time: 66.94s, average batch time: 16.73s
epoch 47 | training: loss: 0.5774, acc: 0.7532, auc:  0.7234 | testing: loss: 0.5566, acc: 0.7492, auc:  0.7029 | time: 67.37s, average batch time: 16.84s
epoch 48 | training: loss: 0.5681, acc: 0.7537, auc:  0.7170 | testing: loss: 0.5882, acc: 0.7540, auc:  0.7361 | time: 67.10s, average batch time: 16.77s
epoch 49 | training: loss: 0.5741, acc: 0.7549, auc:  0.7195 | testing: loss: 0.5659, acc: 0.7661, auc:  0.7340 | time: 67.61s, average batch time: 16.90s
epoch 50 | training: loss: 0.5704, acc: 0.7627, auc:  0.7325 | testing: loss: 0.5604, acc: 0.7548, auc:  0.7196 | time: 67.32s, average batch time: 16.83s
epoch 45 - 50 | training: loss: 0.5727, acc: 0.7555, auc:  0.7241 | testing: loss: 0.5639, acc: 0.7553, auc:  0.7173
epoch 51 | training: loss: 0.5647, acc: 0.7593, auc:  0.7224 | testing: loss: 0.5749, acc: 0.7593, auc:  0.7353 | time: 67.73s, average batch time: 16.93s
epoch 52 | training: loss: 0.5748, acc: 0.7566, auc:  0.7269 | testing: loss: 0.5454, acc: 0.7681, auc:  0.7338 | time: 67.27s, average batch time: 16.82s
epoch 53 | training: loss: 0.5726, acc: 0.7573, auc:  0.7302 | testing: loss: 0.5499, acc: 0.7742, auc:  0.7424 | time: 67.30s, average batch time: 16.83s
epoch 54 | training: loss: 0.5701, acc: 0.7684, auc:  0.7409 | testing: loss: 0.5751, acc: 0.7507, auc:  0.7063 | time: 67.23s, average batch time: 16.81s
epoch 55 | training: loss: 0.5712, acc: 0.7618, auc:  0.7298 | testing: loss: 0.5577, acc: 0.7545, auc:  0.7246 | time: 67.27s, average batch time: 16.82s
epoch 50 - 55 | training: loss: 0.5707, acc: 0.7607, auc:  0.7300 | testing: loss: 0.5606, acc: 0.7614, auc:  0.7285
epoch 56 | training: loss: 0.5674, acc: 0.7650, auc:  0.7323 | testing: loss: 0.5752, acc: 0.7474, auc:  0.7226 | time: 67.79s, average batch time: 16.95s
epoch 57 | training: loss: 0.5688, acc: 0.7674, auc:  0.7353 | testing: loss: 0.5548, acc: 0.7551, auc:  0.7144 | time: 67.44s, average batch time: 16.86s
epoch 58 | training: loss: 0.5684, acc: 0.7631, auc:  0.7264 | testing: loss: 0.5539, acc: 0.7724, auc:  0.7342 | time: 67.71s, average batch time: 16.93s
epoch 59 | training: loss: 0.5725, acc: 0.7624, auc:  0.7319 | testing: loss: 0.5474, acc: 0.7719, auc:  0.7274 | time: 67.27s, average batch time: 16.82s
epoch 60 | training: loss: 0.5606, acc: 0.7674, auc:  0.7347 | testing: loss: 0.5774, acc: 0.7663, auc:  0.7408 | time: 67.46s, average batch time: 16.86s
epoch 55 - 60 | training: loss: 0.5675, acc: 0.7650, auc:  0.7322 | testing: loss: 0.5617, acc: 0.7626, auc:  0.7279
epoch 61 | training: loss: 0.5683, acc: 0.7678, auc:  0.7380 | testing: loss: 0.5570, acc: 0.7554, auc:  0.7208 | time: 67.01s, average batch time: 16.75s
epoch 62 | training: loss: 0.5658, acc: 0.7683, auc:  0.7416 | testing: loss: 0.5550, acc: 0.7619, auc:  0.7220 | time: 66.74s, average batch time: 16.68s
epoch 63 | training: loss: 0.5666, acc: 0.7655, auc:  0.7355 | testing: loss: 0.5489, acc: 0.7922, auc:  0.7658 | time: 67.67s, average batch time: 16.92s
epoch 64 | training: loss: 0.5641, acc: 0.7707, auc:  0.7404 | testing: loss: 0.5619, acc: 0.7765, auc:  0.7395 | time: 67.49s, average batch time: 16.87s
epoch 65 | training: loss: 0.5667, acc: 0.7724, auc:  0.7321 | testing: loss: 0.5659, acc: 0.7798, auc:  0.7433 | time: 67.20s, average batch time: 16.80s
epoch 60 - 65 | training: loss: 0.5663, acc: 0.7690, auc:  0.7375 | testing: loss: 0.5577, acc: 0.7731, auc:  0.7383
epoch 66 | training: loss: 0.5695, acc: 0.7673, auc:  0.7369 | testing: loss: 0.5446, acc: 0.7901, auc:  0.7420 | time: 67.21s, average batch time: 16.80s
epoch 67 | training: loss: 0.5624, acc: 0.7778, auc:  0.7507 | testing: loss: 0.5619, acc: 0.7697, auc:  0.7283 | time: 73.97s, average batch time: 18.49s
epoch 68 | training: loss: 0.5596, acc: 0.7834, auc:  0.7497 | testing: loss: 0.5721, acc: 0.7451, auc:  0.7093 | time: 73.96s, average batch time: 18.49s
epoch 69 | training: loss: 0.5684, acc: 0.7675, auc:  0.7342 | testing: loss: 0.5531, acc: 0.7808, auc:  0.7564 | time: 69.29s, average batch time: 17.32s
epoch 70 | training: loss: 0.5622, acc: 0.7752, auc:  0.7421 | testing: loss: 0.5463, acc: 0.7807, auc:  0.7537 | time: 72.53s, average batch time: 18.13s
epoch 65 - 70 | training: loss: 0.5644, acc: 0.7743, auc:  0.7427 | testing: loss: 0.5556, acc: 0.7733, auc:  0.7379
epoch 71 | training: loss: 0.5668, acc: 0.7732, auc:  0.7404 | testing: loss: 0.5581, acc: 0.7756, auc:  0.7495 | time: 73.10s, average batch time: 18.27s
epoch 72 | training: loss: 0.5649, acc: 0.7804, auc:  0.7535 | testing: loss: 0.5504, acc: 0.7871, auc:  0.7367 | time: 69.97s, average batch time: 17.49s
epoch 73 | training: loss: 0.5562, acc: 0.7845, auc:  0.7552 | testing: loss: 0.5660, acc: 0.7573, auc:  0.7222 | time: 69.45s, average batch time: 17.36s
epoch 74 | training: loss: 0.5593, acc: 0.7750, auc:  0.7387 | testing: loss: 0.5533, acc: 0.7988, auc:  0.7785 | time: 69.26s, average batch time: 17.31s
epoch 75 | training: loss: 0.5642, acc: 0.7816, auc:  0.7531 | testing: loss: 0.5400, acc: 0.7807, auc:  0.7502 | time: 79.59s, average batch time: 19.90s
epoch 70 - 75 | training: loss: 0.5623, acc: 0.7789, auc:  0.7482 | testing: loss: 0.5536, acc: 0.7799, auc:  0.7474
epoch 76 | training: loss: 0.5607, acc: 0.7788, auc:  0.7527 | testing: loss: 0.5515, acc: 0.7691, auc:  0.7472 | time: 76.73s, average batch time: 19.18s
epoch 77 | training: loss: 0.5607, acc: 0.7736, auc:  0.7456 | testing: loss: 0.5519, acc: 0.7877, auc:  0.7580 | time: 76.90s, average batch time: 19.22s
epoch 78 | training: loss: 0.5632, acc: 0.7760, auc:  0.7512 | testing: loss: 0.5523, acc: 0.7775, auc:  0.7385 | time: 86.03s, average batch time: 21.51s
epoch 79 | training: loss: 0.5633, acc: 0.7738, auc:  0.7525 | testing: loss: 0.5441, acc: 0.7686, auc:  0.7305 | time: 79.06s, average batch time: 19.77s
epoch 80 | training: loss: 0.5584, acc: 0.7854, auc:  0.7544 | testing: loss: 0.5560, acc: 0.7817, auc:  0.7473 | time: 77.96s, average batch time: 19.49s
epoch 75 - 80 | training: loss: 0.5613, acc: 0.7775, auc:  0.7513 | testing: loss: 0.5512, acc: 0.7769, auc:  0.7443
epoch 81 | training: loss: 0.5575, acc: 0.7857, auc:  0.7551 | testing: loss: 0.5554, acc: 0.7853, auc:  0.7538 | time: 79.70s, average batch time: 19.93s
epoch 82 | training: loss: 0.5577, acc: 0.7849, auc:  0.7522 | testing: loss: 0.5469, acc: 0.8105, auc:  0.7825 | time: 81.88s, average batch time: 20.47s
epoch 83 | training: loss: 0.5587, acc: 0.7851, auc:  0.7537 | testing: loss: 0.5372, acc: 0.7924, auc:  0.7649 | time: 79.72s, average batch time: 19.93s
epoch 84 | training: loss: 0.5613, acc: 0.7825, auc:  0.7582 | testing: loss: 0.5332, acc: 0.7862, auc:  0.7428 | time: 81.23s, average batch time: 20.31s
epoch 85 | training: loss: 0.5600, acc: 0.7829, auc:  0.7532 | testing: loss: 0.5577, acc: 0.7759, auc:  0.7551 | time: 76.63s, average batch time: 19.16s
epoch 80 - 85 | training: loss: 0.5590, acc: 0.7842, auc:  0.7545 | testing: loss: 0.5461, acc: 0.7901, auc:  0.7598
epoch 86 | training: loss: 0.5549, acc: 0.7841, auc:  0.7600 | testing: loss: 0.5613, acc: 0.7772, auc:  0.7455 | time: 74.16s, average batch time: 18.54s
epoch 87 | training: loss: 0.5630, acc: 0.7876, auc:  0.7621 | testing: loss: 0.5220, acc: 0.8065, auc:  0.7634 | time: 80.85s, average batch time: 20.21s
epoch 88 | training: loss: 0.5561, acc: 0.7856, auc:  0.7561 | testing: loss: 0.5621, acc: 0.7856, auc:  0.7641 | time: 84.51s, average batch time: 21.13s
epoch 89 | training: loss: 0.5533, acc: 0.7846, auc:  0.7497 | testing: loss: 0.5504, acc: 0.7978, auc:  0.7781 | time: 74.61s, average batch time: 18.65s
epoch 90 | training: loss: 0.5565, acc: 0.7929, auc:  0.7638 | testing: loss: 0.5485, acc: 0.7704, auc:  0.7338 | time: 74.32s, average batch time: 18.58s
epoch 85 - 90 | training: loss: 0.5567, acc: 0.7870, auc:  0.7583 | testing: loss: 0.5489, acc: 0.7875, auc:  0.7570
epoch 91 | training: loss: 0.5564, acc: 0.7874, auc:  0.7565 | testing: loss: 0.5613, acc: 0.7772, auc:  0.7404 | time: 83.29s, average batch time: 20.82s
epoch 92 | training: loss: 0.5618, acc: 0.7796, auc:  0.7498 | testing: loss: 0.5386, acc: 0.8051, auc:  0.7728 | time: 81.46s, average batch time: 20.37s
epoch 93 | training: loss: 0.5563, acc: 0.7922, auc:  0.7632 | testing: loss: 0.5535, acc: 0.7612, auc:  0.7266 | time: 79.80s, average batch time: 19.95s
epoch 94 | training: loss: 0.5527, acc: 0.7892, auc:  0.7547 | testing: loss: 0.5458, acc: 0.7840, auc:  0.7575 | time: 74.91s, average batch time: 18.73s
epoch 95 | training: loss: 0.5623, acc: 0.7780, auc:  0.7467 | testing: loss: 0.5370, acc: 0.8246, auc:  0.7893 | time: 81.67s, average batch time: 20.42s
epoch 90 - 95 | training: loss: 0.5579, acc: 0.7853, auc:  0.7542 | testing: loss: 0.5473, acc: 0.7904, auc:  0.7573
epoch 96 | training: loss: 0.5593, acc: 0.7878, auc:  0.7626 | testing: loss: 0.5401, acc: 0.7918, auc:  0.7652 | time: 71.42s, average batch time: 17.86s
epoch 97 | training: loss: 0.5586, acc: 0.7840, auc:  0.7556 | testing: loss: 0.5367, acc: 0.7920, auc:  0.7651 | time: 72.85s, average batch time: 18.21s
epoch 98 | training: loss: 0.5549, acc: 0.7894, auc:  0.7621 | testing: loss: 0.5593, acc: 0.7888, auc:  0.7688 | time: 76.16s, average batch time: 19.04s
epoch 99 | training: loss: 0.5478, acc: 0.7895, auc:  0.7601 | testing: loss: 0.5687, acc: 0.7586, auc:  0.7284 | time: 76.37s, average batch time: 19.09s
epoch 100 | training: loss: 0.5596, acc: 0.7838, auc:  0.7655 | testing: loss: 0.5304, acc: 0.7964, auc:  0.7523 | time: 76.15s, average batch time: 19.04s
epoch 95 - 100 | training: loss: 0.5560, acc: 0.7869, auc:  0.7612 | testing: loss: 0.5470, acc: 0.7855, auc:  0.7559
epoch 101 | training: loss: 0.5572, acc: 0.7884, auc:  0.7679 | testing: loss: 0.5375, acc: 0.7898, auc:  0.7458 | time: 76.37s, average batch time: 19.09s
epoch 102 | training: loss: 0.5609, acc: 0.7819, auc:  0.7543 | testing: loss: 0.5517, acc: 0.7841, auc:  0.7470 | time: 75.62s, average batch time: 18.91s
epoch 103 | training: loss: 0.5537, acc: 0.7875, auc:  0.7591 | testing: loss: 0.5568, acc: 0.7814, auc:  0.7594 | time: 76.33s, average batch time: 19.08s
epoch 104 | training: loss: 0.5577, acc: 0.7897, auc:  0.7642 | testing: loss: 0.5501, acc: 0.7777, auc:  0.7385 | time: 77.09s, average batch time: 19.27s
epoch 105 | training: loss: 0.5591, acc: 0.7798, auc:  0.7459 | testing: loss: 0.5457, acc: 0.8014, auc:  0.7783 | time: 77.87s, average batch time: 19.47s
epoch 100 - 105 | training: loss: 0.5577, acc: 0.7854, auc:  0.7583 | testing: loss: 0.5484, acc: 0.7869, auc:  0.7538
epoch 106 | training: loss: 0.5571, acc: 0.7879, auc:  0.7576 | testing: loss: 0.5461, acc: 0.7939, auc:  0.7669 | time: 76.90s, average batch time: 19.22s
epoch 107 | training: loss: 0.5591, acc: 0.7924, auc:  0.7639 | testing: loss: 0.5473, acc: 0.7702, auc:  0.7301 | time: 77.24s, average batch time: 19.31s
epoch 108 | training: loss: 0.5567, acc: 0.7899, auc:  0.7622 | testing: loss: 0.5399, acc: 0.7910, auc:  0.7531 | time: 75.19s, average batch time: 18.80s
epoch 109 | training: loss: 0.5534, acc: 0.7866, auc:  0.7546 | testing: loss: 0.5507, acc: 0.7965, auc:  0.7639 | time: 78.86s, average batch time: 19.71s
epoch 110 | training: loss: 0.5571, acc: 0.7855, auc:  0.7577 | testing: loss: 0.5516, acc: 0.7905, auc:  0.7605 | time: 72.88s, average batch time: 18.22s
epoch 105 - 110 | training: loss: 0.5567, acc: 0.7884, auc:  0.7592 | testing: loss: 0.5471, acc: 0.7884, auc:  0.7549
epoch 111 | training: loss: 0.5553, acc: 0.7925, auc:  0.7646 | testing: loss: 0.5491, acc: 0.7917, auc:  0.7431 | time: 72.26s, average batch time: 18.06s
epoch 112 | training: loss: 0.5586, acc: 0.7908, auc:  0.7603 | testing: loss: 0.5330, acc: 0.8050, auc:  0.7714 | time: 71.91s, average batch time: 17.98s
epoch 113 | training: loss: 0.5577, acc: 0.7944, auc:  0.7635 | testing: loss: 0.5314, acc: 0.7924, auc:  0.7605 | time: 74.75s, average batch time: 18.69s
epoch 114 | training: loss: 0.5540, acc: 0.7976, auc:  0.7653 | testing: loss: 0.5609, acc: 0.7678, auc:  0.7413 | time: 74.23s, average batch time: 18.56s
epoch 115 | training: loss: 0.5556, acc: 0.7883, auc:  0.7585 | testing: loss: 0.5544, acc: 0.8158, auc:  0.7880 | time: 72.64s, average batch time: 18.16s
epoch 110 - 115 | training: loss: 0.5562, acc: 0.7927, auc:  0.7624 | testing: loss: 0.5458, acc: 0.7945, auc:  0.7609
epoch 116 | training: loss: 0.5555, acc: 0.7982, auc:  0.7671 | testing: loss: 0.5419, acc: 0.7800, auc:  0.7304 | time: 73.08s, average batch time: 18.27s
epoch 117 | training: loss: 0.5528, acc: 0.7941, auc:  0.7636 | testing: loss: 0.5532, acc: 0.7962, auc:  0.7714 | time: 72.53s, average batch time: 18.13s
epoch 118 | training: loss: 0.5557, acc: 0.7892, auc:  0.7612 | testing: loss: 0.5459, acc: 0.7959, auc:  0.7619 | time: 73.22s, average batch time: 18.31s
epoch 119 | training: loss: 0.5572, acc: 0.7932, auc:  0.7657 | testing: loss: 0.5505, acc: 0.7908, auc:  0.7614 | time: 72.87s, average batch time: 18.22s
epoch 120 | training: loss: 0.5545, acc: 0.7906, auc:  0.7555 | testing: loss: 0.5438, acc: 0.8094, auc:  0.7857 | time: 72.11s, average batch time: 18.03s
epoch 115 - 120 | training: loss: 0.5551, acc: 0.7931, auc:  0.7626 | testing: loss: 0.5471, acc: 0.7945, auc:  0.7622
epoch 121 | training: loss: 0.5521, acc: 0.7948, auc:  0.7653 | testing: loss: 0.5475, acc: 0.7967, auc:  0.7661 | time: 74.22s, average batch time: 18.55s
epoch 122 | training: loss: 0.5534, acc: 0.7971, auc:  0.7740 | testing: loss: 0.5408, acc: 0.7933, auc:  0.7612 | time: 72.63s, average batch time: 18.16s
epoch 123 | training: loss: 0.5579, acc: 0.7925, auc:  0.7650 | testing: loss: 0.5306, acc: 0.7962, auc:  0.7692 | time: 72.38s, average batch time: 18.10s
epoch 124 | training: loss: 0.5488, acc: 0.8000, auc:  0.7762 | testing: loss: 0.5567, acc: 0.7788, auc:  0.7467 | time: 72.71s, average batch time: 18.18s
epoch 125 | training: loss: 0.5566, acc: 0.7926, auc:  0.7668 | testing: loss: 0.5452, acc: 0.8085, auc:  0.7826 | time: 72.55s, average batch time: 18.14s
epoch 120 - 125 | training: loss: 0.5538, acc: 0.7954, auc:  0.7695 | testing: loss: 0.5442, acc: 0.7947, auc:  0.7652
epoch 126 | training: loss: 0.5557, acc: 0.7876, auc:  0.7590 | testing: loss: 0.5312, acc: 0.8065, auc:  0.7789 | time: 70.63s, average batch time: 17.66s
epoch 127 | training: loss: 0.5574, acc: 0.7935, auc:  0.7652 | testing: loss: 0.5378, acc: 0.8005, auc:  0.7649 | time: 68.96s, average batch time: 17.24s
epoch 128 | training: loss: 0.5492, acc: 0.7949, auc:  0.7621 | testing: loss: 0.5626, acc: 0.7972, auc:  0.7701 | time: 68.83s, average batch time: 17.21s
epoch 129 | training: loss: 0.5514, acc: 0.7973, auc:  0.7742 | testing: loss: 0.5486, acc: 0.7959, auc:  0.7693 | time: 71.39s, average batch time: 17.85s
epoch 130 | training: loss: 0.5521, acc: 0.8066, auc:  0.7748 | testing: loss: 0.5557, acc: 0.7746, auc:  0.7464 | time: 68.85s, average batch time: 17.21s
epoch 125 - 130 | training: loss: 0.5532, acc: 0.7960, auc:  0.7671 | testing: loss: 0.5472, acc: 0.7949, auc:  0.7659
epoch 131 | training: loss: 0.5529, acc: 0.7974, auc:  0.7677 | testing: loss: 0.5531, acc: 0.7935, auc:  0.7651 | time: 69.45s, average batch time: 17.36s
epoch 132 | training: loss: 0.5524, acc: 0.7971, auc:  0.7712 | testing: loss: 0.5316, acc: 0.8069, auc:  0.7743 | time: 68.94s, average batch time: 17.23s
epoch 133 | training: loss: 0.5467, acc: 0.8052, auc:  0.7805 | testing: loss: 0.5495, acc: 0.7856, auc:  0.7609 | time: 68.74s, average batch time: 17.18s
epoch 134 | training: loss: 0.5574, acc: 0.7943, auc:  0.7679 | testing: loss: 0.5299, acc: 0.7807, auc:  0.7320 | time: 68.57s, average batch time: 17.14s
epoch 135 | training: loss: 0.5481, acc: 0.7962, auc:  0.7661 | testing: loss: 0.5549, acc: 0.8057, auc:  0.7921 | time: 68.89s, average batch time: 17.22s
epoch 130 - 135 | training: loss: 0.5515, acc: 0.7980, auc:  0.7707 | testing: loss: 0.5438, acc: 0.7945, auc:  0.7649
epoch 136 | training: loss: 0.5581, acc: 0.7944, auc:  0.7693 | testing: loss: 0.5357, acc: 0.8005, auc:  0.7696 | time: 68.28s, average batch time: 17.07s
epoch 137 | training: loss: 0.5487, acc: 0.8072, auc:  0.7815 | testing: loss: 0.5564, acc: 0.7633, auc:  0.7233 | time: 69.18s, average batch time: 17.29s
epoch 138 | training: loss: 0.5490, acc: 0.7984, auc:  0.7698 | testing: loss: 0.5505, acc: 0.8141, auc:  0.7937 | time: 68.85s, average batch time: 17.21s
epoch 139 | training: loss: 0.5529, acc: 0.7943, auc:  0.7707 | testing: loss: 0.5296, acc: 0.8089, auc:  0.7735 | time: 68.57s, average batch time: 17.14s
epoch 140 | training: loss: 0.5490, acc: 0.8032, auc:  0.7719 | testing: loss: 0.5523, acc: 0.7919, auc:  0.7543 | time: 69.04s, average batch time: 17.26s
epoch 135 - 140 | training: loss: 0.5515, acc: 0.7995, auc:  0.7726 | testing: loss: 0.5449, acc: 0.7957, auc:  0.7629
epoch 141 | training: loss: 0.5466, acc: 0.7955, auc:  0.7595 | testing: loss: 0.5509, acc: 0.8004, auc:  0.7775 | time: 68.64s, average batch time: 17.16s
epoch 142 | training: loss: 0.5528, acc: 0.8006, auc:  0.7712 | testing: loss: 0.5484, acc: 0.8099, auc:  0.7770 | time: 69.05s, average batch time: 17.26s
epoch 143 | training: loss: 0.5584, acc: 0.7958, auc:  0.7734 | testing: loss: 0.5220, acc: 0.8129, auc:  0.7671 | time: 69.24s, average batch time: 17.31s
epoch 144 | training: loss: 0.5563, acc: 0.8013, auc:  0.7736 | testing: loss: 0.5355, acc: 0.7953, auc:  0.7651 | time: 69.05s, average batch time: 17.26s
epoch 145 | training: loss: 0.5469, acc: 0.8036, auc:  0.7743 | testing: loss: 0.5496, acc: 0.7910, auc:  0.7713 | time: 69.25s, average batch time: 17.31s
epoch 140 - 145 | training: loss: 0.5522, acc: 0.7993, auc:  0.7704 | testing: loss: 0.5413, acc: 0.8019, auc:  0.7716
epoch 146 | training: loss: 0.5498, acc: 0.8010, auc:  0.7710 | testing: loss: 0.5422, acc: 0.7969, auc:  0.7676 | time: 68.69s, average batch time: 17.17s
epoch 147 | training: loss: 0.5513, acc: 0.8047, auc:  0.7779 | testing: loss: 0.5408, acc: 0.8057, auc:  0.7815 | time: 69.22s, average batch time: 17.31s
epoch 148 | training: loss: 0.5506, acc: 0.7944, auc:  0.7700 | testing: loss: 0.5457, acc: 0.8022, auc:  0.7713 | time: 69.06s, average batch time: 17.26s
epoch 149 | training: loss: 0.5535, acc: 0.8003, auc:  0.7709 | testing: loss: 0.5501, acc: 0.7923, auc:  0.7606 | time: 69.17s, average batch time: 17.29s
epoch 150 | training: loss: 0.5543, acc: 0.7966, auc:  0.7718 | testing: loss: 0.5331, acc: 0.8028, auc:  0.7639 | time: 69.37s, average batch time: 17.34s
epoch 145 - 150 | training: loss: 0.5519, acc: 0.7994, auc:  0.7723 | testing: loss: 0.5424, acc: 0.8000, auc:  0.7690
epoch 151 | training: loss: 0.5545, acc: 0.7977, auc:  0.7706 | testing: loss: 0.5482, acc: 0.7876, auc:  0.7526 | time: 69.59s, average batch time: 17.40s
epoch 152 | training: loss: 0.5475, acc: 0.8020, auc:  0.7751 | testing: loss: 0.5622, acc: 0.7893, auc:  0.7604 | time: 68.74s, average batch time: 17.19s
epoch 153 | training: loss: 0.5520, acc: 0.7966, auc:  0.7664 | testing: loss: 0.5412, acc: 0.7957, auc:  0.7689 | time: 69.32s, average batch time: 17.33s
epoch 154 | training: loss: 0.5493, acc: 0.8027, auc:  0.7769 | testing: loss: 0.5429, acc: 0.7960, auc:  0.7620 | time: 68.96s, average batch time: 17.24s
epoch 155 | training: loss: 0.5543, acc: 0.7957, auc:  0.7727 | testing: loss: 0.5347, acc: 0.8032, auc:  0.7656 | time: 71.81s, average batch time: 17.95s
epoch 150 - 155 | training: loss: 0.5515, acc: 0.7989, auc:  0.7723 | testing: loss: 0.5458, acc: 0.7944, auc:  0.7619
epoch 156 | training: loss: 0.5587, acc: 0.7942, auc:  0.7700 | testing: loss: 0.5169, acc: 0.8102, auc:  0.7649 | time: 69.60s, average batch time: 17.40s
epoch 157 | training: loss: 0.5509, acc: 0.8075, auc:  0.7729 | testing: loss: 0.5421, acc: 0.7969, auc:  0.7743 | time: 68.68s, average batch time: 17.17s
epoch 158 | training: loss: 0.5490, acc: 0.8052, auc:  0.7818 | testing: loss: 0.5453, acc: 0.7818, auc:  0.7482 | time: 69.02s, average batch time: 17.25s
epoch 159 | training: loss: 0.5476, acc: 0.8018, auc:  0.7769 | testing: loss: 0.5466, acc: 0.7898, auc:  0.7504 | time: 69.21s, average batch time: 17.30s
epoch 160 | training: loss: 0.5460, acc: 0.7992, auc:  0.7713 | testing: loss: 0.5450, acc: 0.8270, auc:  0.8032 | time: 69.98s, average batch time: 17.50s
epoch 155 - 160 | training: loss: 0.5505, acc: 0.8016, auc:  0.7746 | testing: loss: 0.5392, acc: 0.8011, auc:  0.7682
epoch 161 | training: loss: 0.5471, acc: 0.8051, auc:  0.7739 | testing: loss: 0.5577, acc: 0.7905, auc:  0.7715 | time: 68.74s, average batch time: 17.19s
epoch 162 | training: loss: 0.5465, acc: 0.8014, auc:  0.7711 | testing: loss: 0.5541, acc: 0.8066, auc:  0.7836 | time: 68.41s, average batch time: 17.10s
epoch 163 | training: loss: 0.5523, acc: 0.8049, auc:  0.7843 | testing: loss: 0.5264, acc: 0.8040, auc:  0.7623 | time: 68.89s, average batch time: 17.22s
epoch 164 | training: loss: 0.5535, acc: 0.7984, auc:  0.7710 | testing: loss: 0.5234, acc: 0.8162, auc:  0.7809 | time: 69.72s, average batch time: 17.43s
epoch 165 | training: loss: 0.5498, acc: 0.8091, auc:  0.7812 | testing: loss: 0.5401, acc: 0.7890, auc:  0.7659 | time: 69.00s, average batch time: 17.25s
epoch 160 - 165 | training: loss: 0.5499, acc: 0.8038, auc:  0.7763 | testing: loss: 0.5403, acc: 0.8012, auc:  0.7728
epoch 166 | training: loss: 0.5482, acc: 0.8011, auc:  0.7764 | testing: loss: 0.5416, acc: 0.8071, auc:  0.7720 | time: 68.89s, average batch time: 17.22s
epoch 167 | training: loss: 0.5502, acc: 0.8016, auc:  0.7775 | testing: loss: 0.5300, acc: 0.8091, auc:  0.7655 | time: 69.16s, average batch time: 17.29s
epoch 168 | training: loss: 0.5456, acc: 0.8025, auc:  0.7753 | testing: loss: 0.5534, acc: 0.8111, auc:  0.7909 | time: 69.38s, average batch time: 17.34s
epoch 169 | training: loss: 0.5565, acc: 0.7972, auc:  0.7707 | testing: loss: 0.5349, acc: 0.7962, auc:  0.7623 | time: 68.80s, average batch time: 17.20s
epoch 170 | training: loss: 0.5477, acc: 0.8077, auc:  0.7812 | testing: loss: 0.5365, acc: 0.8019, auc:  0.7795 | time: 69.15s, average batch time: 17.29s
epoch 165 - 170 | training: loss: 0.5496, acc: 0.8020, auc:  0.7762 | testing: loss: 0.5393, acc: 0.8051, auc:  0.7740
epoch 171 | training: loss: 0.5535, acc: 0.8025, auc:  0.7774 | testing: loss: 0.5320, acc: 0.8063, auc:  0.7675 | time: 68.74s, average batch time: 17.19s
epoch 172 | training: loss: 0.5483, acc: 0.7999, auc:  0.7745 | testing: loss: 0.5474, acc: 0.8151, auc:  0.7804 | time: 68.65s, average batch time: 17.16s
epoch 173 | training: loss: 0.5505, acc: 0.8044, auc:  0.7744 | testing: loss: 0.5441, acc: 0.7947, auc:  0.7659 | time: 70.57s, average batch time: 17.64s
epoch 174 | training: loss: 0.5511, acc: 0.8020, auc:  0.7779 | testing: loss: 0.5340, acc: 0.7982, auc:  0.7716 | time: 74.18s, average batch time: 18.55s
epoch 175 | training: loss: 0.5438, acc: 0.8028, auc:  0.7708 | testing: loss: 0.5504, acc: 0.7925, auc:  0.7781 | time: 69.27s, average batch time: 17.32s
epoch 170 - 175 | training: loss: 0.5494, acc: 0.8023, auc:  0.7750 | testing: loss: 0.5416, acc: 0.8014, auc:  0.7727
epoch 176 | training: loss: 0.5505, acc: 0.8057, auc:  0.7797 | testing: loss: 0.5504, acc: 0.7855, auc:  0.7493 | time: 69.06s, average batch time: 17.26s
epoch 177 | training: loss: 0.5483, acc: 0.8013, auc:  0.7812 | testing: loss: 0.5416, acc: 0.8109, auc:  0.7826 | time: 68.98s, average batch time: 17.24s
epoch 178 | training: loss: 0.5505, acc: 0.8075, auc:  0.7863 | testing: loss: 0.5444, acc: 0.7721, auc:  0.7417 | time: 68.39s, average batch time: 17.10s
epoch 179 | training: loss: 0.5447, acc: 0.7996, auc:  0.7722 | testing: loss: 0.5393, acc: 0.8144, auc:  0.7857 | time: 69.02s, average batch time: 17.25s
epoch 180 | training: loss: 0.5498, acc: 0.8013, auc:  0.7716 | testing: loss: 0.5405, acc: 0.8117, auc:  0.7914 | time: 68.99s, average batch time: 17.25s
epoch 175 - 180 | training: loss: 0.5488, acc: 0.8031, auc:  0.7782 | testing: loss: 0.5432, acc: 0.7989, auc:  0.7701
epoch 181 | training: loss: 0.5479, acc: 0.8090, auc:  0.7825 | testing: loss: 0.5386, acc: 0.8135, auc:  0.7796 | time: 69.70s, average batch time: 17.43s
epoch 182 | training: loss: 0.5488, acc: 0.8060, auc:  0.7813 | testing: loss: 0.5422, acc: 0.8104, auc:  0.7876 | time: 69.18s, average batch time: 17.29s
epoch 183 | training: loss: 0.5474, acc: 0.8068, auc:  0.7776 | testing: loss: 0.5452, acc: 0.7950, auc:  0.7622 | time: 68.70s, average batch time: 17.18s
epoch 184 | training: loss: 0.5445, acc: 0.8085, auc:  0.7838 | testing: loss: 0.5571, acc: 0.7715, auc:  0.7412 | time: 69.06s, average batch time: 17.26s
epoch 185 | training: loss: 0.5553, acc: 0.7948, auc:  0.7730 | testing: loss: 0.5328, acc: 0.8143, auc:  0.7815 | time: 69.25s, average batch time: 17.31s
epoch 180 - 185 | training: loss: 0.5488, acc: 0.8050, auc:  0.7796 | testing: loss: 0.5432, acc: 0.8010, auc:  0.7704
epoch 186 | training: loss: 0.5467, acc: 0.8026, auc:  0.7808 | testing: loss: 0.5429, acc: 0.8000, auc:  0.7754 | time: 68.97s, average batch time: 17.24s
epoch 187 | training: loss: 0.5499, acc: 0.7995, auc:  0.7719 | testing: loss: 0.5452, acc: 0.8000, auc:  0.7755 | time: 69.39s, average batch time: 17.35s
epoch 188 | training: loss: 0.5521, acc: 0.8051, auc:  0.7737 | testing: loss: 0.5432, acc: 0.8040, auc:  0.7700 | time: 69.40s, average batch time: 17.35s
epoch 189 | training: loss: 0.5542, acc: 0.7991, auc:  0.7732 | testing: loss: 0.5325, acc: 0.8092, auc:  0.7629 | time: 68.80s, average batch time: 17.20s
epoch 190 | training: loss: 0.5492, acc: 0.8011, auc:  0.7786 | testing: loss: 0.5366, acc: 0.8018, auc:  0.7633 | time: 69.00s, average batch time: 17.25s
epoch 185 - 190 | training: loss: 0.5504, acc: 0.8015, auc:  0.7756 | testing: loss: 0.5401, acc: 0.8030, auc:  0.7694
epoch 191 | training: loss: 0.5520, acc: 0.7993, auc:  0.7738 | testing: loss: 0.5441, acc: 0.8070, auc:  0.7611 | time: 68.56s, average batch time: 17.14s
epoch 192 | training: loss: 0.5502, acc: 0.8022, auc:  0.7757 | testing: loss: 0.5417, acc: 0.8153, auc:  0.7884 | time: 68.96s, average batch time: 17.24s
epoch 193 | training: loss: 0.5440, acc: 0.8008, auc:  0.7689 | testing: loss: 0.5390, acc: 0.8242, auc:  0.7999 | time: 69.19s, average batch time: 17.30s
epoch 194 | training: loss: 0.5506, acc: 0.8094, auc:  0.7909 | testing: loss: 0.5335, acc: 0.7835, auc:  0.7389 | time: 69.82s, average batch time: 17.46s
epoch 195 | training: loss: 0.5509, acc: 0.8081, auc:  0.7793 | testing: loss: 0.5432, acc: 0.7989, auc:  0.7780 | time: 69.27s, average batch time: 17.32s
epoch 190 - 195 | training: loss: 0.5495, acc: 0.8040, auc:  0.7777 | testing: loss: 0.5403, acc: 0.8058, auc:  0.7733
epoch 196 | training: loss: 0.5446, acc: 0.8068, auc:  0.7825 | testing: loss: 0.5580, acc: 0.7910, auc:  0.7722 | time: 68.95s, average batch time: 17.24s
epoch 197 | training: loss: 0.5471, acc: 0.8030, auc:  0.7750 | testing: loss: 0.5437, acc: 0.8127, auc:  0.7955 | time: 69.16s, average batch time: 17.29s
epoch 198 | training: loss: 0.5485, acc: 0.8043, auc:  0.7774 | testing: loss: 0.5453, acc: 0.7956, auc:  0.7620 | time: 68.76s, average batch time: 17.19s
epoch 199 | training: loss: 0.5568, acc: 0.8014, auc:  0.7762 | testing: loss: 0.5295, acc: 0.8124, auc:  0.7816 | time: 69.08s, average batch time: 17.27s
epoch 200 | training: loss: 0.5506, acc: 0.8073, auc:  0.7819 | testing: loss: 0.5344, acc: 0.8024, auc:  0.7638 | time: 68.87s, average batch time: 17.22s
epoch 195 - 200 | training: loss: 0.5495, acc: 0.8045, auc:  0.7786 | testing: loss: 0.5422, acc: 0.8028, auc:  0.7750
epoch 201 | training: loss: 0.5450, acc: 0.8058, auc:  0.7796 | testing: loss: 0.5575, acc: 0.7967, auc:  0.7750 | time: 68.91s, average batch time: 17.23s
epoch 202 | training: loss: 0.5507, acc: 0.8005, auc:  0.7729 | testing: loss: 0.5398, acc: 0.8172, auc:  0.7826 | time: 68.58s, average batch time: 17.14s
epoch 203 | training: loss: 0.5528, acc: 0.8045, auc:  0.7798 | testing: loss: 0.5174, acc: 0.8087, auc:  0.7701 | time: 69.30s, average batch time: 17.33s
epoch 204 | training: loss: 0.5524, acc: 0.8079, auc:  0.7826 | testing: loss: 0.5299, acc: 0.8101, auc:  0.7812 | time: 69.02s, average batch time: 17.25s
epoch 205 | training: loss: 0.5437, acc: 0.8047, auc:  0.7843 | testing: loss: 0.5581, acc: 0.7906, auc:  0.7657 | time: 69.23s, average batch time: 17.31s
epoch 200 - 205 | training: loss: 0.5489, acc: 0.8047, auc:  0.7798 | testing: loss: 0.5406, acc: 0.8047, auc:  0.7749
epoch 206 | training: loss: 0.5464, acc: 0.8143, auc:  0.7875 | testing: loss: 0.5552, acc: 0.7880, auc:  0.7673 | time: 68.69s, average batch time: 17.17s
epoch 207 | training: loss: 0.5495, acc: 0.8011, auc:  0.7779 | testing: loss: 0.5280, acc: 0.8155, auc:  0.7781 | time: 69.94s, average batch time: 17.49s
epoch 208 | training: loss: 0.5491, acc: 0.7979, auc:  0.7803 | testing: loss: 0.5370, acc: 0.8159, auc:  0.7762 | time: 69.01s, average batch time: 17.25s
epoch 209 | training: loss: 0.5478, acc: 0.8050, auc:  0.7795 | testing: loss: 0.5366, acc: 0.8124, auc:  0.7823 | time: 68.83s, average batch time: 17.21s
epoch 210 | training: loss: 0.5484, acc: 0.8091, auc:  0.7814 | testing: loss: 0.5436, acc: 0.8066, auc:  0.7819 | time: 68.97s, average batch time: 17.24s
epoch 205 - 210 | training: loss: 0.5482, acc: 0.8055, auc:  0.7813 | testing: loss: 0.5401, acc: 0.8077, auc:  0.7772
epoch 211 | training: loss: 0.5457, acc: 0.8034, auc:  0.7806 | testing: loss: 0.5410, acc: 0.8008, auc:  0.7793 | time: 68.92s, average batch time: 17.23s
epoch 212 | training: loss: 0.5495, acc: 0.8071, auc:  0.7840 | testing: loss: 0.5298, acc: 0.8038, auc:  0.7587 | time: 69.64s, average batch time: 17.41s
epoch 213 | training: loss: 0.5524, acc: 0.7939, auc:  0.7719 | testing: loss: 0.5335, acc: 0.8223, auc:  0.7993 | time: 68.85s, average batch time: 17.21s
epoch 214 | training: loss: 0.5413, acc: 0.8105, auc:  0.7904 | testing: loss: 0.5577, acc: 0.7951, auc:  0.7728 | time: 70.25s, average batch time: 17.56s
epoch 215 | training: loss: 0.5496, acc: 0.8081, auc:  0.7761 | testing: loss: 0.5331, acc: 0.8083, auc:  0.7695 | time: 68.87s, average batch time: 17.22s
epoch 210 - 215 | training: loss: 0.5477, acc: 0.8046, auc:  0.7806 | testing: loss: 0.5390, acc: 0.8061, auc:  0.7759
epoch 216 | training: loss: 0.5463, acc: 0.8096, auc:  0.7869 | testing: loss: 0.5437, acc: 0.8004, auc:  0.7777 | time: 69.80s, average batch time: 17.45s
epoch 217 | training: loss: 0.5468, acc: 0.8058, auc:  0.7868 | testing: loss: 0.5408, acc: 0.8023, auc:  0.7599 | time: 68.46s, average batch time: 17.12s
epoch 218 | training: loss: 0.5449, acc: 0.8051, auc:  0.7786 | testing: loss: 0.5422, acc: 0.8166, auc:  0.8051 | time: 68.74s, average batch time: 17.18s
epoch 219 | training: loss: 0.5512, acc: 0.8048, auc:  0.7866 | testing: loss: 0.5361, acc: 0.7964, auc:  0.7581 | time: 68.80s, average batch time: 17.20s
epoch 220 | training: loss: 0.5471, acc: 0.8075, auc:  0.7820 | testing: loss: 0.5315, acc: 0.8183, auc:  0.7895 | time: 69.77s, average batch time: 17.44s
epoch 215 - 220 | training: loss: 0.5472, acc: 0.8065, auc:  0.7842 | testing: loss: 0.5388, acc: 0.8068, auc:  0.7781
epoch 221 | training: loss: 0.5460, acc: 0.8060, auc:  0.7812 | testing: loss: 0.5252, acc: 0.8189, auc:  0.7976 | time: 69.13s, average batch time: 17.28s
epoch 222 | training: loss: 0.5490, acc: 0.8105, auc:  0.7914 | testing: loss: 0.5342, acc: 0.8056, auc:  0.7624 | time: 68.89s, average batch time: 17.22s
epoch 223 | training: loss: 0.5457, acc: 0.8125, auc:  0.7870 | testing: loss: 0.5457, acc: 0.7970, auc:  0.7703 | time: 69.21s, average batch time: 17.30s
epoch 224 | training: loss: 0.5500, acc: 0.8044, auc:  0.7791 | testing: loss: 0.5345, acc: 0.8108, auc:  0.7801 | time: 68.61s, average batch time: 17.15s
epoch 225 | training: loss: 0.5385, acc: 0.8091, auc:  0.7814 | testing: loss: 0.5576, acc: 0.7990, auc:  0.7764 | time: 69.08s, average batch time: 17.27s
epoch 220 - 225 | training: loss: 0.5458, acc: 0.8085, auc:  0.7840 | testing: loss: 0.5394, acc: 0.8063, auc:  0.7774
epoch 226 | training: loss: 0.5469, acc: 0.8036, auc:  0.7787 | testing: loss: 0.5421, acc: 0.8127, auc:  0.7972 | time: 68.62s, average batch time: 17.16s
epoch 227 | training: loss: 0.5444, acc: 0.8033, auc:  0.7742 | testing: loss: 0.5329, acc: 0.8254, auc:  0.8130 | time: 69.38s, average batch time: 17.35s
epoch 228 | training: loss: 0.5460, acc: 0.8075, auc:  0.7835 | testing: loss: 0.5485, acc: 0.7946, auc:  0.7677 | time: 68.91s, average batch time: 17.23s
epoch 229 | training: loss: 0.5452, acc: 0.8078, auc:  0.7872 | testing: loss: 0.5483, acc: 0.8053, auc:  0.7771 | time: 69.09s, average batch time: 17.27s
epoch 230 | training: loss: 0.5507, acc: 0.8073, auc:  0.7906 | testing: loss: 0.5203, acc: 0.8063, auc:  0.7672 | time: 68.96s, average batch time: 17.24s
epoch 225 - 230 | training: loss: 0.5466, acc: 0.8059, auc:  0.7829 | testing: loss: 0.5384, acc: 0.8089, auc:  0.7844
epoch 231 | training: loss: 0.5477, acc: 0.8100, auc:  0.7894 | testing: loss: 0.5316, acc: 0.8108, auc:  0.7776 | time: 70.03s, average batch time: 17.51s
epoch 232 | training: loss: 0.5433, acc: 0.8076, auc:  0.7856 | testing: loss: 0.5365, acc: 0.8317, auc:  0.8041 | time: 85.61s, average batch time: 21.40s
epoch 233 | training: loss: 0.5484, acc: 0.8079, auc:  0.7900 | testing: loss: 0.5360, acc: 0.8083, auc:  0.7783 | time: 78.44s, average batch time: 19.61s
epoch 234 | training: loss: 0.5450, acc: 0.8118, auc:  0.7859 | testing: loss: 0.5368, acc: 0.7954, auc:  0.7727 | time: 74.85s, average batch time: 18.71s
epoch 235 | training: loss: 0.5460, acc: 0.8136, auc:  0.7850 | testing: loss: 0.5411, acc: 0.8046, auc:  0.7839 | time: 76.86s, average batch time: 19.22s
epoch 230 - 235 | training: loss: 0.5461, acc: 0.8102, auc:  0.7872 | testing: loss: 0.5364, acc: 0.8102, auc:  0.7833
epoch 236 | training: loss: 0.5437, acc: 0.8122, auc:  0.7848 | testing: loss: 0.5469, acc: 0.8061, auc:  0.7817 | time: 74.60s, average batch time: 18.65s
epoch 237 | training: loss: 0.5440, acc: 0.8097, auc:  0.7899 | testing: loss: 0.5391, acc: 0.8071, auc:  0.7735 | time: 74.84s, average batch time: 18.71s
epoch 238 | training: loss: 0.5428, acc: 0.8141, auc:  0.7904 | testing: loss: 0.5350, acc: 0.8128, auc:  0.7947 | time: 77.66s, average batch time: 19.41s
epoch 239 | training: loss: 0.5434, acc: 0.8067, auc:  0.7822 | testing: loss: 0.5335, acc: 0.8232, auc:  0.7942 | time: 81.77s, average batch time: 20.44s
epoch 240 | training: loss: 0.5484, acc: 0.8118, auc:  0.7906 | testing: loss: 0.5231, acc: 0.8160, auc:  0.7864 | time: 75.41s, average batch time: 18.85s
epoch 235 - 240 | training: loss: 0.5445, acc: 0.8109, auc:  0.7876 | testing: loss: 0.5355, acc: 0.8130, auc:  0.7861
epoch 241 | training: loss: 0.5415, acc: 0.8116, auc:  0.7886 | testing: loss: 0.5309, acc: 0.7993, auc:  0.7654 | time: 78.62s, average batch time: 19.66s
epoch 242 | training: loss: 0.5396, acc: 0.8134, auc:  0.7913 | testing: loss: 0.5500, acc: 0.8026, auc:  0.7783 | time: 77.34s, average batch time: 19.34s
epoch 243 | training: loss: 0.5446, acc: 0.8062, auc:  0.7836 | testing: loss: 0.5456, acc: 0.8029, auc:  0.7860 | time: 95.70s, average batch time: 23.92s
epoch 244 | training: loss: 0.5490, acc: 0.8111, auc:  0.7918 | testing: loss: 0.5306, acc: 0.8076, auc:  0.7806 | time: 80.71s, average batch time: 20.18s
epoch 245 | training: loss: 0.5480, acc: 0.8022, auc:  0.7847 | testing: loss: 0.5200, acc: 0.8328, auc:  0.8056 | time: 75.55s, average batch time: 18.89s
epoch 240 - 245 | training: loss: 0.5446, acc: 0.8089, auc:  0.7880 | testing: loss: 0.5354, acc: 0.8090, auc:  0.7832
epoch 246 | training: loss: 0.5458, acc: 0.8085, auc:  0.7843 | testing: loss: 0.5447, acc: 0.8131, auc:  0.7907 | time: 75.56s, average batch time: 18.89s
epoch 247 | training: loss: 0.5450, acc: 0.8165, auc:  0.7923 | testing: loss: 0.5373, acc: 0.8057, auc:  0.7891 | time: 83.97s, average batch time: 20.99s
epoch 248 | training: loss: 0.5439, acc: 0.8094, auc:  0.7912 | testing: loss: 0.5330, acc: 0.8113, auc:  0.7739 | time: 90.49s, average batch time: 22.62s
epoch 249 | training: loss: 0.5415, acc: 0.8129, auc:  0.7915 | testing: loss: 0.5341, acc: 0.8017, auc:  0.7725 | time: 87.65s, average batch time: 21.91s
epoch 250 | training: loss: 0.5403, acc: 0.8185, auc:  0.8029 | testing: loss: 0.5230, acc: 0.8257, auc:  0.7903 | time: 81.76s, average batch time: 20.44s
epoch 245 - 250 | training: loss: 0.5433, acc: 0.8132, auc:  0.7924 | testing: loss: 0.5344, acc: 0.8115, auc:  0.7833
epoch 251 | training: loss: 0.5441, acc: 0.8126, auc:  0.7916 | testing: loss: 0.5431, acc: 0.8106, auc:  0.7918 | time: 84.15s, average batch time: 21.04s
epoch 252 | training: loss: 0.5503, acc: 0.8089, auc:  0.7933 | testing: loss: 0.5217, acc: 0.8035, auc:  0.7720 | time: 82.27s, average batch time: 20.57s
epoch 253 | training: loss: 0.5408, acc: 0.8124, auc:  0.7926 | testing: loss: 0.5401, acc: 0.7937, auc:  0.7720 | time: 78.94s, average batch time: 19.73s
epoch 254 | training: loss: 0.5409, acc: 0.8110, auc:  0.7888 | testing: loss: 0.5375, acc: 0.8169, auc:  0.7932 | time: 76.32s, average batch time: 19.08s
epoch 255 | training: loss: 0.5408, acc: 0.8113, auc:  0.7917 | testing: loss: 0.5461, acc: 0.8151, auc:  0.7854 | time: 83.29s, average batch time: 20.82s
epoch 250 - 255 | training: loss: 0.5434, acc: 0.8112, auc:  0.7916 | testing: loss: 0.5377, acc: 0.8080, auc:  0.7829
epoch 256 | training: loss: 0.5452, acc: 0.8113, auc:  0.7924 | testing: loss: 0.5293, acc: 0.7978, auc:  0.7574 | time: 76.90s, average batch time: 19.23s
epoch 257 | training: loss: 0.5458, acc: 0.8105, auc:  0.7894 | testing: loss: 0.5303, acc: 0.8006, auc:  0.7793 | time: 74.21s, average batch time: 18.55s
epoch 258 | training: loss: 0.5419, acc: 0.8120, auc:  0.7910 | testing: loss: 0.5458, acc: 0.8027, auc:  0.7780 | time: 73.23s, average batch time: 18.31s
epoch 259 | training: loss: 0.5470, acc: 0.8052, auc:  0.7836 | testing: loss: 0.5277, acc: 0.8375, auc:  0.8119 | time: 73.32s, average batch time: 18.33s
epoch 260 | training: loss: 0.5418, acc: 0.8174, auc:  0.7874 | testing: loss: 0.5467, acc: 0.8079, auc:  0.7874 | time: 74.81s, average batch time: 18.70s
epoch 255 - 260 | training: loss: 0.5443, acc: 0.8113, auc:  0.7888 | testing: loss: 0.5360, acc: 0.8093, auc:  0.7828
epoch 261 | training: loss: 0.5386, acc: 0.8167, auc:  0.7900 | testing: loss: 0.5537, acc: 0.8023, auc:  0.7909 | time: 74.52s, average batch time: 18.63s
epoch 262 | training: loss: 0.5425, acc: 0.8104, auc:  0.7919 | testing: loss: 0.5284, acc: 0.8133, auc:  0.7878 | time: 73.96s, average batch time: 18.49s
epoch 263 | training: loss: 0.5487, acc: 0.8060, auc:  0.7847 | testing: loss: 0.5260, acc: 0.8238, auc:  0.7944 | time: 74.60s, average batch time: 18.65s
epoch 264 | training: loss: 0.5478, acc: 0.8106, auc:  0.7888 | testing: loss: 0.5308, acc: 0.8104, auc:  0.7877 | time: 74.15s, average batch time: 18.54s
epoch 265 | training: loss: 0.5408, acc: 0.8171, auc:  0.7992 | testing: loss: 0.5249, acc: 0.8039, auc:  0.7665 | time: 74.59s, average batch time: 18.65s
epoch 260 - 265 | training: loss: 0.5437, acc: 0.8122, auc:  0.7909 | testing: loss: 0.5327, acc: 0.8107, auc:  0.7855
epoch 266 | training: loss: 0.5426, acc: 0.8104, auc:  0.7921 | testing: loss: 0.5361, acc: 0.8238, auc:  0.7949 | time: 74.03s, average batch time: 18.51s
epoch 267 | training: loss: 0.5380, acc: 0.8080, auc:  0.7928 | testing: loss: 0.5596, acc: 0.7996, auc:  0.7830 | time: 74.54s, average batch time: 18.63s
epoch 268 | training: loss: 0.5489, acc: 0.8080, auc:  0.7907 | testing: loss: 0.5324, acc: 0.8066, auc:  0.7750 | time: 74.67s, average batch time: 18.67s
epoch 269 | training: loss: 0.5432, acc: 0.8122, auc:  0.7919 | testing: loss: 0.5310, acc: 0.8132, auc:  0.8031 | time: 74.40s, average batch time: 18.60s
epoch 270 | training: loss: 0.5495, acc: 0.8104, auc:  0.7951 | testing: loss: 0.5231, acc: 0.8170, auc:  0.7855 | time: 74.15s, average batch time: 18.54s
epoch 265 - 270 | training: loss: 0.5444, acc: 0.8098, auc:  0.7925 | testing: loss: 0.5364, acc: 0.8120, auc:  0.7883
epoch 271 | training: loss: 0.5460, acc: 0.8096, auc:  0.7908 | testing: loss: 0.5239, acc: 0.8095, auc:  0.7688 | time: 73.91s, average batch time: 18.48s
epoch 272 | training: loss: 0.5410, acc: 0.8159, auc:  0.7937 | testing: loss: 0.5539, acc: 0.8021, auc:  0.7871 | time: 73.93s, average batch time: 18.48s
epoch 273 | training: loss: 0.5400, acc: 0.8133, auc:  0.7966 | testing: loss: 0.5405, acc: 0.8077, auc:  0.7751 | time: 73.93s, average batch time: 18.48s
epoch 274 | training: loss: 0.5439, acc: 0.8062, auc:  0.7865 | testing: loss: 0.5206, acc: 0.8292, auc:  0.8070 | time: 73.63s, average batch time: 18.41s
epoch 275 | training: loss: 0.5472, acc: 0.8038, auc:  0.7890 | testing: loss: 0.5378, acc: 0.8122, auc:  0.7960 | time: 74.18s, average batch time: 18.54s
epoch 270 - 275 | training: loss: 0.5436, acc: 0.8098, auc:  0.7913 | testing: loss: 0.5353, acc: 0.8122, auc:  0.7868
epoch 276 | training: loss: 0.5407, acc: 0.8044, auc:  0.7831 | testing: loss: 0.5520, acc: 0.8080, auc:  0.7853 | time: 73.60s, average batch time: 18.40s
epoch 277 | training: loss: 0.5458, acc: 0.8138, auc:  0.7948 | testing: loss: 0.5395, acc: 0.8036, auc:  0.7835 | time: 73.84s, average batch time: 18.46s
epoch 278 | training: loss: 0.5445, acc: 0.8091, auc:  0.7858 | testing: loss: 0.5383, acc: 0.8017, auc:  0.7812 | time: 73.91s, average batch time: 18.48s
epoch 279 | training: loss: 0.5455, acc: 0.8048, auc:  0.7925 | testing: loss: 0.5262, acc: 0.8193, auc:  0.7753 | time: 74.95s, average batch time: 18.74s
epoch 280 | training: loss: 0.5462, acc: 0.8095, auc:  0.7897 | testing: loss: 0.5323, acc: 0.8047, auc:  0.7730 | time: 74.67s, average batch time: 18.67s
epoch 275 - 280 | training: loss: 0.5445, acc: 0.8083, auc:  0.7892 | testing: loss: 0.5377, acc: 0.8075, auc:  0.7796
epoch 281 | training: loss: 0.5388, acc: 0.8149, auc:  0.7916 | testing: loss: 0.5417, acc: 0.8138, auc:  0.7925 | time: 74.67s, average batch time: 18.67s
epoch 282 | training: loss: 0.5452, acc: 0.8062, auc:  0.7882 | testing: loss: 0.5202, acc: 0.8280, auc:  0.8069 | time: 74.17s, average batch time: 18.54s
epoch 283 | training: loss: 0.5462, acc: 0.8095, auc:  0.7921 | testing: loss: 0.5335, acc: 0.8098, auc:  0.7803 | time: 74.03s, average batch time: 18.51s
epoch 284 | training: loss: 0.5423, acc: 0.8103, auc:  0.7944 | testing: loss: 0.5381, acc: 0.7965, auc:  0.7763 | time: 73.88s, average batch time: 18.47s
epoch 285 | training: loss: 0.5452, acc: 0.8051, auc:  0.7882 | testing: loss: 0.5438, acc: 0.8011, auc:  0.7753 | time: 74.62s, average batch time: 18.65s
epoch 280 - 285 | training: loss: 0.5435, acc: 0.8092, auc:  0.7909 | testing: loss: 0.5355, acc: 0.8099, auc:  0.7862
epoch 286 | training: loss: 0.5425, acc: 0.8109, auc:  0.7930 | testing: loss: 0.5404, acc: 0.8093, auc:  0.7837 | time: 74.13s, average batch time: 18.53s
epoch 287 | training: loss: 0.5406, acc: 0.8155, auc:  0.7921 | testing: loss: 0.5468, acc: 0.7937, auc:  0.7677 | time: 74.32s, average batch time: 18.58s
epoch 288 | training: loss: 0.5434, acc: 0.8050, auc:  0.7855 | testing: loss: 0.5315, acc: 0.8307, auc:  0.8106 | time: 74.16s, average batch time: 18.54s
epoch 289 | training: loss: 0.5461, acc: 0.8145, auc:  0.7999 | testing: loss: 0.5312, acc: 0.8010, auc:  0.7608 | time: 74.00s, average batch time: 18.50s
epoch 290 | training: loss: 0.5463, acc: 0.8057, auc:  0.7860 | testing: loss: 0.5402, acc: 0.8068, auc:  0.7769 | time: 73.54s, average batch time: 18.38s
epoch 285 - 290 | training: loss: 0.5438, acc: 0.8103, auc:  0.7913 | testing: loss: 0.5380, acc: 0.8083, auc:  0.7799
epoch 291 | training: loss: 0.5439, acc: 0.8088, auc:  0.7910 | testing: loss: 0.5326, acc: 0.8206, auc:  0.8012 | time: 74.86s, average batch time: 18.72s
epoch 292 | training: loss: 0.5470, acc: 0.8104, auc:  0.7899 | testing: loss: 0.5120, acc: 0.8308, auc:  0.7958 | time: 74.53s, average batch time: 18.63s
epoch 293 | training: loss: 0.5448, acc: 0.8134, auc:  0.7952 | testing: loss: 0.5452, acc: 0.8056, auc:  0.7727 | time: 74.34s, average batch time: 18.59s
epoch 294 | training: loss: 0.5386, acc: 0.8156, auc:  0.7925 | testing: loss: 0.5599, acc: 0.7946, auc:  0.7693 | time: 73.97s, average batch time: 18.49s
epoch 295 | training: loss: 0.5435, acc: 0.8082, auc:  0.7856 | testing: loss: 0.5425, acc: 0.8131, auc:  0.7897 | time: 74.00s, average batch time: 18.50s
epoch 290 - 295 | training: loss: 0.5436, acc: 0.8113, auc:  0.7908 | testing: loss: 0.5385, acc: 0.8129, auc:  0.7857
epoch 296 | training: loss: 0.5401, acc: 0.8211, auc:  0.8005 | testing: loss: 0.5344, acc: 0.8067, auc:  0.7790 | time: 74.51s, average batch time: 18.63s
epoch 297 | training: loss: 0.5443, acc: 0.8113, auc:  0.7904 | testing: loss: 0.5312, acc: 0.8186, auc:  0.7966 | time: 73.79s, average batch time: 18.45s
epoch 298 | training: loss: 0.5439, acc: 0.8166, auc:  0.7965 | testing: loss: 0.5505, acc: 0.7864, auc:  0.7441 | time: 74.07s, average batch time: 18.52s
epoch 299 | training: loss: 0.5408, acc: 0.8142, auc:  0.7962 | testing: loss: 0.5406, acc: 0.8086, auc:  0.7910 | time: 73.50s, average batch time: 18.38s
epoch 300 | training: loss: 0.5451, acc: 0.8061, auc:  0.7860 | testing: loss: 0.5191, acc: 0.8391, auc:  0.8121 | time: 74.25s, average batch time: 18.56s
epoch 295 - 300 | training: loss: 0.5429, acc: 0.8138, auc:  0.7939 | testing: loss: 0.5352, acc: 0.8119, auc:  0.7846
epoch 301 | training: loss: 0.5376, acc: 0.8168, auc:  0.7942 | testing: loss: 0.5539, acc: 0.8105, auc:  0.7978 | time: 73.51s, average batch time: 18.38s
epoch 302 | training: loss: 0.5493, acc: 0.8139, auc:  0.7989 | testing: loss: 0.5090, acc: 0.8246, auc:  0.7718 | time: 76.56s, average batch time: 19.14s
epoch 303 | training: loss: 0.5423, acc: 0.8148, auc:  0.7974 | testing: loss: 0.5207, acc: 0.8204, auc:  0.7902 | time: 80.11s, average batch time: 20.03s
epoch 304 | training: loss: 0.5384, acc: 0.8193, auc:  0.7992 | testing: loss: 0.5505, acc: 0.8020, auc:  0.7736 | time: 83.82s, average batch time: 20.95s
epoch 305 | training: loss: 0.5396, acc: 0.8145, auc:  0.7903 | testing: loss: 0.5370, acc: 0.8328, auc:  0.8116 | time: 91.53s, average batch time: 22.88s
epoch 300 - 305 | training: loss: 0.5414, acc: 0.8159, auc:  0.7960 | testing: loss: 0.5342, acc: 0.8180, auc:  0.7890
epoch 306 | training: loss: 0.5390, acc: 0.8173, auc:  0.7952 | testing: loss: 0.5335, acc: 0.8081, auc:  0.7808 | time: 82.26s, average batch time: 20.57s
epoch 307 | training: loss: 0.5390, acc: 0.8121, auc:  0.7952 | testing: loss: 0.5351, acc: 0.8106, auc:  0.7944 | time: 88.07s, average batch time: 22.02s
epoch 308 | training: loss: 0.5431, acc: 0.8131, auc:  0.7960 | testing: loss: 0.5378, acc: 0.8152, auc:  0.7915 | time: 79.48s, average batch time: 19.87s
epoch 309 | training: loss: 0.5439, acc: 0.8126, auc:  0.7960 | testing: loss: 0.5375, acc: 0.8088, auc:  0.7745 | time: 76.47s, average batch time: 19.12s
epoch 310 | training: loss: 0.5443, acc: 0.8100, auc:  0.7935 | testing: loss: 0.5303, acc: 0.8188, auc:  0.7929 | time: 77.88s, average batch time: 19.47s
epoch 305 - 310 | training: loss: 0.5419, acc: 0.8130, auc:  0.7952 | testing: loss: 0.5348, acc: 0.8123, auc:  0.7868
epoch 311 | training: loss: 0.5417, acc: 0.8124, auc:  0.7942 | testing: loss: 0.5329, acc: 0.8221, auc:  0.8062 | time: 79.81s, average batch time: 19.95s
epoch 312 | training: loss: 0.5400, acc: 0.8140, auc:  0.7937 | testing: loss: 0.5349, acc: 0.8193, auc:  0.7909 | time: 76.70s, average batch time: 19.17s
epoch 313 | training: loss: 0.5388, acc: 0.8232, auc:  0.8037 | testing: loss: 0.5345, acc: 0.8089, auc:  0.7730 | time: 75.48s, average batch time: 18.87s
epoch 314 | training: loss: 0.5455, acc: 0.8130, auc:  0.7924 | testing: loss: 0.5226, acc: 0.8444, auc:  0.8146 | time: 78.01s, average batch time: 19.50s
epoch 315 | training: loss: 0.5422, acc: 0.8131, auc:  0.7933 | testing: loss: 0.5375, acc: 0.8166, auc:  0.7826 | time: 77.30s, average batch time: 19.32s
epoch 310 - 315 | training: loss: 0.5416, acc: 0.8151, auc:  0.7955 | testing: loss: 0.5325, acc: 0.8223, auc:  0.7934
epoch 316 | training: loss: 0.5453, acc: 0.8171, auc:  0.7976 | testing: loss: 0.5258, acc: 0.8130, auc:  0.7829 | time: 77.19s, average batch time: 19.30s
epoch 317 | training: loss: 0.5372, acc: 0.8194, auc:  0.7989 | testing: loss: 0.5358, acc: 0.8296, auc:  0.7990 | time: 76.92s, average batch time: 19.23s
epoch 318 | training: loss: 0.5487, acc: 0.8070, auc:  0.7855 | testing: loss: 0.5278, acc: 0.8248, auc:  0.7947 | time: 74.56s, average batch time: 18.64s
epoch 319 | training: loss: 0.5350, acc: 0.8223, auc:  0.8013 | testing: loss: 0.5428, acc: 0.8116, auc:  0.7964 | time: 79.30s, average batch time: 19.82s
epoch 320 | training: loss: 0.5411, acc: 0.8170, auc:  0.7977 | testing: loss: 0.5373, acc: 0.8032, auc:  0.7721 | time: 78.52s, average batch time: 19.63s
epoch 315 - 320 | training: loss: 0.5415, acc: 0.8166, auc:  0.7962 | testing: loss: 0.5339, acc: 0.8164, auc:  0.7890
epoch 321 | training: loss: 0.5392, acc: 0.8211, auc:  0.8011 | testing: loss: 0.5340, acc: 0.8160, auc:  0.7935 | time: 77.45s, average batch time: 19.36s
epoch 322 | training: loss: 0.5439, acc: 0.8167, auc:  0.7942 | testing: loss: 0.5315, acc: 0.8198, auc:  0.7897 | time: 77.85s, average batch time: 19.46s
epoch 323 | training: loss: 0.5394, acc: 0.8190, auc:  0.7979 | testing: loss: 0.5390, acc: 0.8099, auc:  0.7804 | time: 77.35s, average batch time: 19.34s
epoch 324 | training: loss: 0.5434, acc: 0.8191, auc:  0.7999 | testing: loss: 0.5319, acc: 0.8088, auc:  0.7773 | time: 78.71s, average batch time: 19.68s
epoch 325 | training: loss: 0.5440, acc: 0.8112, auc:  0.7827 | testing: loss: 0.5249, acc: 0.8354, auc:  0.8205 | time: 80.29s, average batch time: 20.07s
epoch 320 - 325 | training: loss: 0.5420, acc: 0.8174, auc:  0.7951 | testing: loss: 0.5323, acc: 0.8180, auc:  0.7923
epoch 326 | training: loss: 0.5388, acc: 0.8195, auc:  0.7998 | testing: loss: 0.5299, acc: 0.8183, auc:  0.7917 | time: 84.80s, average batch time: 21.20s
epoch 327 | training: loss: 0.5405, acc: 0.8171, auc:  0.7986 | testing: loss: 0.5393, acc: 0.8051, auc:  0.7776 | time: 90.18s, average batch time: 22.55s
epoch 328 | training: loss: 0.5427, acc: 0.8179, auc:  0.7976 | testing: loss: 0.5337, acc: 0.8168, auc:  0.7950 | time: 83.67s, average batch time: 20.92s
epoch 329 | training: loss: 0.5483, acc: 0.8075, auc:  0.7867 | testing: loss: 0.5229, acc: 0.8273, auc:  0.8055 | time: 82.49s, average batch time: 20.62s
epoch 330 | training: loss: 0.5400, acc: 0.8142, auc:  0.7998 | testing: loss: 0.5509, acc: 0.8020, auc:  0.7781 | time: 79.66s, average batch time: 19.92s
epoch 325 - 330 | training: loss: 0.5420, acc: 0.8152, auc:  0.7965 | testing: loss: 0.5353, acc: 0.8139, auc:  0.7896
epoch 331 | training: loss: 0.5437, acc: 0.8163, auc:  0.7911 | testing: loss: 0.5458, acc: 0.8144, auc:  0.7966 | time: 74.70s, average batch time: 18.68s
epoch 332 | training: loss: 0.5444, acc: 0.8100, auc:  0.7917 | testing: loss: 0.5347, acc: 0.8133, auc:  0.7901 | time: 74.78s, average batch time: 18.70s
epoch 333 | training: loss: 0.5401, acc: 0.8174, auc:  0.7959 | testing: loss: 0.5440, acc: 0.8006, auc:  0.7735 | time: 74.64s, average batch time: 18.66s
epoch 334 | training: loss: 0.5422, acc: 0.8111, auc:  0.7960 | testing: loss: 0.5274, acc: 0.8293, auc:  0.8005 | time: 74.61s, average batch time: 18.65s
epoch 335 | training: loss: 0.5416, acc: 0.8171, auc:  0.8007 | testing: loss: 0.5221, acc: 0.8170, auc:  0.7881 | time: 74.73s, average batch time: 18.68s
epoch 330 - 335 | training: loss: 0.5424, acc: 0.8144, auc:  0.7951 | testing: loss: 0.5348, acc: 0.8149, auc:  0.7898
epoch 336 | training: loss: 0.5470, acc: 0.8070, auc:  0.7865 | testing: loss: 0.5244, acc: 0.8196, auc:  0.7914 | time: 74.37s, average batch time: 18.59s
epoch 337 | training: loss: 0.5326, acc: 0.8292, auc:  0.8074 | testing: loss: 0.5428, acc: 0.8102, auc:  0.7732 | time: 74.78s, average batch time: 18.69s
epoch 338 | training: loss: 0.5398, acc: 0.8180, auc:  0.7970 | testing: loss: 0.5351, acc: 0.8183, auc:  0.7971 | time: 74.47s, average batch time: 18.62s
epoch 339 | training: loss: 0.5429, acc: 0.8141, auc:  0.7947 | testing: loss: 0.5288, acc: 0.8117, auc:  0.7829 | time: 74.75s, average batch time: 18.69s
epoch 340 | training: loss: 0.5398, acc: 0.8168, auc:  0.7966 | testing: loss: 0.5258, acc: 0.8297, auc:  0.8080 | time: 74.80s, average batch time: 18.70s
epoch 335 - 340 | training: loss: 0.5404, acc: 0.8170, auc:  0.7964 | testing: loss: 0.5314, acc: 0.8179, auc:  0.7905
epoch 341 | training: loss: 0.5419, acc: 0.8186, auc:  0.7980 | testing: loss: 0.5292, acc: 0.8191, auc:  0.8006 | time: 74.92s, average batch time: 18.73s
epoch 342 | training: loss: 0.5466, acc: 0.8159, auc:  0.7964 | testing: loss: 0.5169, acc: 0.8370, auc:  0.8107 | time: 74.86s, average batch time: 18.72s
epoch 343 | training: loss: 0.5366, acc: 0.8205, auc:  0.7937 | testing: loss: 0.5487, acc: 0.8108, auc:  0.7862 | time: 74.39s, average batch time: 18.60s
epoch 344 | training: loss: 0.5406, acc: 0.8146, auc:  0.7945 | testing: loss: 0.5310, acc: 0.8189, auc:  0.7964 | time: 74.93s, average batch time: 18.73s
epoch 345 | training: loss: 0.5390, acc: 0.8197, auc:  0.7999 | testing: loss: 0.5386, acc: 0.8095, auc:  0.7663 | time: 75.72s, average batch time: 18.93s
epoch 340 - 345 | training: loss: 0.5409, acc: 0.8179, auc:  0.7965 | testing: loss: 0.5329, acc: 0.8190, auc:  0.7920
epoch 346 | training: loss: 0.5478, acc: 0.8138, auc:  0.7935 | testing: loss: 0.5142, acc: 0.8279, auc:  0.7981 | time: 74.61s, average batch time: 18.65s
epoch 347 | training: loss: 0.5345, acc: 0.8258, auc:  0.8055 | testing: loss: 0.5353, acc: 0.8120, auc:  0.7876 | time: 74.88s, average batch time: 18.72s
epoch 348 | training: loss: 0.5430, acc: 0.8159, auc:  0.7968 | testing: loss: 0.5243, acc: 0.8282, auc:  0.7991 | time: 74.44s, average batch time: 18.61s
epoch 349 | training: loss: 0.5370, acc: 0.8207, auc:  0.7958 | testing: loss: 0.5505, acc: 0.8174, auc:  0.7971 | time: 75.99s, average batch time: 19.00s
epoch 350 | training: loss: 0.5397, acc: 0.8154, auc:  0.7962 | testing: loss: 0.5470, acc: 0.8158, auc:  0.7939 | time: 75.35s, average batch time: 18.84s
epoch 345 - 350 | training: loss: 0.5404, acc: 0.8183, auc:  0.7976 | testing: loss: 0.5343, acc: 0.8203, auc:  0.7952
epoch 351 | training: loss: 0.5423, acc: 0.8174, auc:  0.7923 | testing: loss: 0.5363, acc: 0.8277, auc:  0.8120 | time: 74.66s, average batch time: 18.67s
epoch 352 | training: loss: 0.5456, acc: 0.8154, auc:  0.7984 | testing: loss: 0.5401, acc: 0.8256, auc:  0.7927 | time: 74.69s, average batch time: 18.67s
epoch 353 | training: loss: 0.5458, acc: 0.8116, auc:  0.7858 | testing: loss: 0.5293, acc: 0.8106, auc:  0.7847 | time: 74.50s, average batch time: 18.62s
epoch 354 | training: loss: 0.5366, acc: 0.8185, auc:  0.7954 | testing: loss: 0.5391, acc: 0.8185, auc:  0.7892 | time: 74.35s, average batch time: 18.59s
epoch 355 | training: loss: 0.5419, acc: 0.8135, auc:  0.7973 | testing: loss: 0.5320, acc: 0.8214, auc:  0.7958 | time: 74.09s, average batch time: 18.52s
epoch 350 - 355 | training: loss: 0.5425, acc: 0.8153, auc:  0.7938 | testing: loss: 0.5354, acc: 0.8207, auc:  0.7949
epoch 356 | training: loss: 0.5427, acc: 0.8161, auc:  0.7929 | testing: loss: 0.5354, acc: 0.8035, auc:  0.7811 | time: 75.11s, average batch time: 18.78s
epoch 357 | training: loss: 0.5350, acc: 0.8219, auc:  0.8069 | testing: loss: 0.5487, acc: 0.8027, auc:  0.7846 | time: 74.36s, average batch time: 18.59s
epoch 358 | training: loss: 0.5434, acc: 0.8146, auc:  0.7962 | testing: loss: 0.5284, acc: 0.8222, auc:  0.8046 | time: 74.80s, average batch time: 18.70s
epoch 359 | training: loss: 0.5459, acc: 0.8107, auc:  0.7914 | testing: loss: 0.5285, acc: 0.8234, auc:  0.8037 | time: 74.38s, average batch time: 18.59s
epoch 360 | training: loss: 0.5390, acc: 0.8179, auc:  0.7987 | testing: loss: 0.5324, acc: 0.8135, auc:  0.7813 | time: 74.85s, average batch time: 18.71s
epoch 355 - 360 | training: loss: 0.5412, acc: 0.8163, auc:  0.7972 | testing: loss: 0.5347, acc: 0.8131, auc:  0.7910
epoch 361 | training: loss: 0.5443, acc: 0.8136, auc:  0.7913 | testing: loss: 0.5293, acc: 0.8347, auc:  0.8178 | time: 75.04s, average batch time: 18.76s
epoch 362 | training: loss: 0.5456, acc: 0.8111, auc:  0.7923 | testing: loss: 0.5224, acc: 0.8228, auc:  0.7910 | time: 74.61s, average batch time: 18.65s
epoch 363 | training: loss: 0.5463, acc: 0.8113, auc:  0.7892 | testing: loss: 0.5298, acc: 0.8215, auc:  0.7920 | time: 75.09s, average batch time: 18.77s
epoch 364 | training: loss: 0.5380, acc: 0.8201, auc:  0.7981 | testing: loss: 0.5474, acc: 0.8055, auc:  0.7900 | time: 74.77s, average batch time: 18.69s
epoch 365 | training: loss: 0.5369, acc: 0.8251, auc:  0.8060 | testing: loss: 0.5346, acc: 0.8000, auc:  0.7774 | time: 74.59s, average batch time: 18.65s
epoch 360 - 365 | training: loss: 0.5422, acc: 0.8162, auc:  0.7954 | testing: loss: 0.5327, acc: 0.8169, auc:  0.7936
epoch 366 | training: loss: 0.5469, acc: 0.8146, auc:  0.7937 | testing: loss: 0.5217, acc: 0.8260, auc:  0.7926 | time: 74.53s, average batch time: 18.63s
epoch 367 | training: loss: 0.5364, acc: 0.8178, auc:  0.7959 | testing: loss: 0.5515, acc: 0.8086, auc:  0.7874 | time: 74.82s, average batch time: 18.71s
epoch 368 | training: loss: 0.5438, acc: 0.8148, auc:  0.7948 | testing: loss: 0.5194, acc: 0.8269, auc:  0.8019 | time: 74.39s, average batch time: 18.60s
epoch 369 | training: loss: 0.5430, acc: 0.8192, auc:  0.7952 | testing: loss: 0.5327, acc: 0.8163, auc:  0.7878 | time: 74.86s, average batch time: 18.71s
epoch 370 | training: loss: 0.5381, acc: 0.8191, auc:  0.8002 | testing: loss: 0.5326, acc: 0.8243, auc:  0.7962 | time: 74.55s, average batch time: 18.64s
epoch 365 - 370 | training: loss: 0.5416, acc: 0.8171, auc:  0.7960 | testing: loss: 0.5316, acc: 0.8204, auc:  0.7932
epoch 371 | training: loss: 0.5470, acc: 0.8121, auc:  0.7967 | testing: loss: 0.5093, acc: 0.8438, auc:  0.8037 | time: 74.20s, average batch time: 18.55s
epoch 372 | training: loss: 0.5353, acc: 0.8282, auc:  0.8059 | testing: loss: 0.5424, acc: 0.8038, auc:  0.7822 | time: 75.48s, average batch time: 18.87s
epoch 373 | training: loss: 0.5385, acc: 0.8167, auc:  0.7942 | testing: loss: 0.5367, acc: 0.8148, auc:  0.8013 | time: 75.37s, average batch time: 18.84s
epoch 374 | training: loss: 0.5366, acc: 0.8168, auc:  0.7974 | testing: loss: 0.5264, acc: 0.8356, auc:  0.8018 | time: 75.16s, average batch time: 18.79s
epoch 375 | training: loss: 0.5421, acc: 0.8176, auc:  0.7979 | testing: loss: 0.5404, acc: 0.8103, auc:  0.7806 | time: 74.69s, average batch time: 18.67s
epoch 370 - 375 | training: loss: 0.5399, acc: 0.8183, auc:  0.7984 | testing: loss: 0.5310, acc: 0.8217, auc:  0.7939
epoch 376 | training: loss: 0.5346, acc: 0.8330, auc:  0.8111 | testing: loss: 0.5505, acc: 0.7856, auc:  0.7597 | time: 74.15s, average batch time: 18.54s
epoch 377 | training: loss: 0.5403, acc: 0.8190, auc:  0.7960 | testing: loss: 0.5222, acc: 0.8382, auc:  0.8083 | time: 75.04s, average batch time: 18.76s
epoch 378 | training: loss: 0.5388, acc: 0.8167, auc:  0.7961 | testing: loss: 0.5413, acc: 0.8287, auc:  0.8067 | time: 74.09s, average batch time: 18.52s
epoch 379 | training: loss: 0.5409, acc: 0.8201, auc:  0.8057 | testing: loss: 0.5271, acc: 0.8211, auc:  0.7785 | time: 74.29s, average batch time: 18.57s
epoch 380 | training: loss: 0.5456, acc: 0.8110, auc:  0.7934 | testing: loss: 0.5167, acc: 0.8355, auc:  0.8115 | time: 74.74s, average batch time: 18.68s
epoch 375 - 380 | training: loss: 0.5400, acc: 0.8200, auc:  0.8005 | testing: loss: 0.5316, acc: 0.8218, auc:  0.7930
epoch 381 | training: loss: 0.5414, acc: 0.8174, auc:  0.7951 | testing: loss: 0.5281, acc: 0.8322, auc:  0.8049 | time: 74.55s, average batch time: 18.64s
epoch 382 | training: loss: 0.5455, acc: 0.8128, auc:  0.7922 | testing: loss: 0.5182, acc: 0.8361, auc:  0.8096 | time: 74.28s, average batch time: 18.57s
epoch 383 | training: loss: 0.5387, acc: 0.8182, auc:  0.7973 | testing: loss: 0.5483, acc: 0.8071, auc:  0.7883 | time: 74.27s, average batch time: 18.57s
epoch 384 | training: loss: 0.5420, acc: 0.8148, auc:  0.7941 | testing: loss: 0.5296, acc: 0.8179, auc:  0.7954 | time: 73.91s, average batch time: 18.48s
epoch 385 | training: loss: 0.5420, acc: 0.8137, auc:  0.7923 | testing: loss: 0.5275, acc: 0.8197, auc:  0.7964 | time: 75.02s, average batch time: 18.75s
epoch 380 - 385 | training: loss: 0.5419, acc: 0.8154, auc:  0.7942 | testing: loss: 0.5304, acc: 0.8226, auc:  0.7989
epoch 386 | training: loss: 0.5405, acc: 0.8252, auc:  0.8055 | testing: loss: 0.5291, acc: 0.8117, auc:  0.7809 | time: 75.64s, average batch time: 18.91s
epoch 387 | training: loss: 0.5376, acc: 0.8174, auc:  0.8008 | testing: loss: 0.5413, acc: 0.8235, auc:  0.8041 | time: 74.94s, average batch time: 18.74s
epoch 388 | training: loss: 0.5447, acc: 0.8106, auc:  0.7921 | testing: loss: 0.5281, acc: 0.8268, auc:  0.8043 | time: 75.60s, average batch time: 18.90s
epoch 389 | training: loss: 0.5410, acc: 0.8212, auc:  0.8019 | testing: loss: 0.5276, acc: 0.8095, auc:  0.7750 | time: 75.36s, average batch time: 18.84s
epoch 390 | training: loss: 0.5393, acc: 0.8179, auc:  0.7951 | testing: loss: 0.5381, acc: 0.8151, auc:  0.7947 | time: 74.85s, average batch time: 18.71s
epoch 385 - 390 | training: loss: 0.5406, acc: 0.8185, auc:  0.7991 | testing: loss: 0.5329, acc: 0.8173, auc:  0.7918
epoch 391 | training: loss: 0.5429, acc: 0.8166, auc:  0.7982 | testing: loss: 0.5264, acc: 0.8261, auc:  0.8022 | time: 74.44s, average batch time: 18.61s
epoch 392 | training: loss: 0.5423, acc: 0.8231, auc:  0.7995 | testing: loss: 0.5437, acc: 0.8142, auc:  0.7931 | time: 74.55s, average batch time: 18.64s
epoch 393 | training: loss: 0.5466, acc: 0.8123, auc:  0.7924 | testing: loss: 0.5248, acc: 0.8236, auc:  0.7984 | time: 73.45s, average batch time: 18.36s
epoch 394 | training: loss: 0.5393, acc: 0.8186, auc:  0.7988 | testing: loss: 0.5397, acc: 0.8026, auc:  0.7691 | time: 76.08s, average batch time: 19.02s
epoch 395 | training: loss: 0.5374, acc: 0.8182, auc:  0.8020 | testing: loss: 0.5482, acc: 0.8000, auc:  0.7729 | time: 75.40s, average batch time: 18.85s
epoch 390 - 395 | training: loss: 0.5417, acc: 0.8178, auc:  0.7982 | testing: loss: 0.5366, acc: 0.8133, auc:  0.7871
epoch 396 | training: loss: 0.5401, acc: 0.8207, auc:  0.7967 | testing: loss: 0.5359, acc: 0.8101, auc:  0.7874 | time: 75.25s, average batch time: 18.81s
epoch 397 | training: loss: 0.5488, acc: 0.8110, auc:  0.7951 | testing: loss: 0.5107, acc: 0.8409, auc:  0.8142 | time: 75.19s, average batch time: 18.80s
epoch 398 | training: loss: 0.5331, acc: 0.8255, auc:  0.8075 | testing: loss: 0.5540, acc: 0.8057, auc:  0.7886 | time: 75.37s, average batch time: 18.84s
epoch 399 | training: loss: 0.5405, acc: 0.8207, auc:  0.8031 | testing: loss: 0.5308, acc: 0.8197, auc:  0.7830 | time: 74.73s, average batch time: 18.68s
epoch 400 | training: loss: 0.5396, acc: 0.8162, auc:  0.7935 | testing: loss: 0.5319, acc: 0.8166, auc:  0.7966 | time: 74.54s, average batch time: 18.63s
epoch 395 - 400 | training: loss: 0.5404, acc: 0.8188, auc:  0.7992 | testing: loss: 0.5327, acc: 0.8186, auc:  0.7940
epoch 401 | training: loss: 0.5445, acc: 0.8190, auc:  0.7991 | testing: loss: 0.5260, acc: 0.8187, auc:  0.7874 | time: 74.72s, average batch time: 18.68s
epoch 402 | training: loss: 0.5392, acc: 0.8232, auc:  0.8031 | testing: loss: 0.5322, acc: 0.8091, auc:  0.7683 | time: 75.20s, average batch time: 18.80s
epoch 403 | training: loss: 0.5367, acc: 0.8144, auc:  0.7915 | testing: loss: 0.5402, acc: 0.8254, auc:  0.8098 | time: 74.30s, average batch time: 18.58s
epoch 404 | training: loss: 0.5438, acc: 0.8137, auc:  0.7948 | testing: loss: 0.5209, acc: 0.8291, auc:  0.8009 | time: 75.34s, average batch time: 18.83s
epoch 405 | training: loss: 0.5392, acc: 0.8230, auc:  0.8015 | testing: loss: 0.5357, acc: 0.8090, auc:  0.7884 | time: 74.27s, average batch time: 18.57s
epoch 400 - 405 | training: loss: 0.5407, acc: 0.8186, auc:  0.7980 | testing: loss: 0.5310, acc: 0.8183, auc:  0.7910
epoch 406 | training: loss: 0.5335, acc: 0.8201, auc:  0.8004 | testing: loss: 0.5561, acc: 0.7936, auc:  0.7760 | time: 74.82s, average batch time: 18.71s
epoch 407 | training: loss: 0.5457, acc: 0.8136, auc:  0.7945 | testing: loss: 0.5253, acc: 0.8157, auc:  0.7919 | time: 74.82s, average batch time: 18.71s
epoch 408 | training: loss: 0.5439, acc: 0.8071, auc:  0.7915 | testing: loss: 0.5132, acc: 0.8400, auc:  0.8075 | time: 74.23s, average batch time: 18.56s
epoch 409 | training: loss: 0.5422, acc: 0.8163, auc:  0.7976 | testing: loss: 0.5318, acc: 0.8142, auc:  0.7907 | time: 74.93s, average batch time: 18.73s
epoch 410 | training: loss: 0.5393, acc: 0.8144, auc:  0.7923 | testing: loss: 0.5384, acc: 0.8135, auc:  0.7968 | time: 75.01s, average batch time: 18.75s
epoch 405 - 410 | training: loss: 0.5409, acc: 0.8143, auc:  0.7953 | testing: loss: 0.5329, acc: 0.8154, auc:  0.7926
epoch 411 | training: loss: 0.5400, acc: 0.8190, auc:  0.8019 | testing: loss: 0.5287, acc: 0.7982, auc:  0.7628 | time: 74.98s, average batch time: 18.75s
epoch 412 | training: loss: 0.5392, acc: 0.8154, auc:  0.7955 | testing: loss: 0.5263, acc: 0.8307, auc:  0.8109 | time: 74.70s, average batch time: 18.67s
epoch 413 | training: loss: 0.5438, acc: 0.8180, auc:  0.7990 | testing: loss: 0.5358, acc: 0.8215, auc:  0.7900 | time: 74.04s, average batch time: 18.51s
epoch 414 | training: loss: 0.5447, acc: 0.8115, auc:  0.7897 | testing: loss: 0.5340, acc: 0.8138, auc:  0.7936 | time: 74.47s, average batch time: 18.62s
epoch 415 | training: loss: 0.5401, acc: 0.8159, auc:  0.7934 | testing: loss: 0.5359, acc: 0.8156, auc:  0.7906 | time: 74.87s, average batch time: 18.72s
epoch 410 - 415 | training: loss: 0.5416, acc: 0.8160, auc:  0.7959 | testing: loss: 0.5321, acc: 0.8160, auc:  0.7896
epoch 416 | training: loss: 0.5426, acc: 0.8134, auc:  0.7961 | testing: loss: 0.5283, acc: 0.8089, auc:  0.7842 | time: 74.47s, average batch time: 18.62s
epoch 417 | training: loss: 0.5411, acc: 0.8093, auc:  0.7871 | testing: loss: 0.5356, acc: 0.8364, auc:  0.8084 | time: 74.53s, average batch time: 18.63s
epoch 418 | training: loss: 0.5402, acc: 0.8239, auc:  0.8062 | testing: loss: 0.5249, acc: 0.8264, auc:  0.7964 | time: 74.55s, average batch time: 18.64s
epoch 419 | training: loss: 0.5443, acc: 0.8196, auc:  0.7995 | testing: loss: 0.5283, acc: 0.8043, auc:  0.7763 | time: 82.47s, average batch time: 20.62s
epoch 420 | training: loss: 0.5410, acc: 0.8207, auc:  0.7959 | testing: loss: 0.5448, acc: 0.8140, auc:  0.7919 | time: 87.24s, average batch time: 21.81s
epoch 415 - 420 | training: loss: 0.5418, acc: 0.8174, auc:  0.7970 | testing: loss: 0.5324, acc: 0.8180, auc:  0.7914
epoch 421 | training: loss: 0.5398, acc: 0.8170, auc:  0.7922 | testing: loss: 0.5353, acc: 0.8332, auc:  0.8096 | time: 83.41s, average batch time: 20.85s
epoch 422 | training: loss: 0.5393, acc: 0.8220, auc:  0.7995 | testing: loss: 0.5363, acc: 0.7961, auc:  0.7769 | time: 82.16s, average batch time: 20.54s
epoch 423 | training: loss: 0.5431, acc: 0.8150, auc:  0.7951 | testing: loss: 0.5346, acc: 0.8178, auc:  0.7809 | time: 79.17s, average batch time: 19.79s
epoch 424 | training: loss: 0.5371, acc: 0.8204, auc:  0.7963 | testing: loss: 0.5379, acc: 0.8201, auc:  0.7963 | time: 79.51s, average batch time: 19.88s
epoch 425 | training: loss: 0.5462, acc: 0.8164, auc:  0.7977 | testing: loss: 0.5282, acc: 0.8077, auc:  0.7768 | time: 80.46s, average batch time: 20.11s
epoch 420 - 425 | training: loss: 0.5411, acc: 0.8182, auc:  0.7961 | testing: loss: 0.5345, acc: 0.8150, auc:  0.7881
epoch 426 | training: loss: 0.5380, acc: 0.8107, auc:  0.7863 | testing: loss: 0.5320, acc: 0.8293, auc:  0.8062 | time: 80.42s, average batch time: 20.11s
epoch 427 | training: loss: 0.5427, acc: 0.8228, auc:  0.7993 | testing: loss: 0.5256, acc: 0.8153, auc:  0.7816 | time: 77.54s, average batch time: 19.39s
epoch 428 | training: loss: 0.5385, acc: 0.8174, auc:  0.7997 | testing: loss: 0.5341, acc: 0.8178, auc:  0.7916 | time: 74.42s, average batch time: 18.60s
epoch 429 | training: loss: 0.5373, acc: 0.8193, auc:  0.7983 | testing: loss: 0.5333, acc: 0.8306, auc:  0.8077 | time: 74.54s, average batch time: 18.63s
epoch 430 | training: loss: 0.5425, acc: 0.8127, auc:  0.7960 | testing: loss: 0.5278, acc: 0.8132, auc:  0.7884 | time: 75.13s, average batch time: 18.78s
epoch 425 - 430 | training: loss: 0.5398, acc: 0.8166, auc:  0.7959 | testing: loss: 0.5306, acc: 0.8212, auc:  0.7951
epoch 431 | training: loss: 0.5326, acc: 0.8242, auc:  0.7976 | testing: loss: 0.5539, acc: 0.8109, auc:  0.7948 | time: 73.97s, average batch time: 18.49s
epoch 432 | training: loss: 0.5379, acc: 0.8210, auc:  0.7948 | testing: loss: 0.5343, acc: 0.8067, auc:  0.7961 | time: 75.12s, average batch time: 18.78s
epoch 433 | training: loss: 0.5393, acc: 0.8126, auc:  0.7999 | testing: loss: 0.5281, acc: 0.8337, auc:  0.7899 | time: 74.78s, average batch time: 18.70s
epoch 434 | training: loss: 0.5436, acc: 0.8162, auc:  0.7989 | testing: loss: 0.5200, acc: 0.8187, auc:  0.7814 | time: 74.69s, average batch time: 18.67s
epoch 435 | training: loss: 0.5464, acc: 0.8145, auc:  0.7940 | testing: loss: 0.5259, acc: 0.8197, auc:  0.7838 | time: 74.27s, average batch time: 18.57s
epoch 430 - 435 | training: loss: 0.5400, acc: 0.8177, auc:  0.7970 | testing: loss: 0.5324, acc: 0.8180, auc:  0.7892
epoch 436 | training: loss: 0.5383, acc: 0.8152, auc:  0.7996 | testing: loss: 0.5485, acc: 0.8027, auc:  0.7769 | time: 74.50s, average batch time: 18.63s
epoch 437 | training: loss: 0.5402, acc: 0.8136, auc:  0.7879 | testing: loss: 0.5299, acc: 0.8155, auc:  0.7914 | time: 75.76s, average batch time: 18.94s
epoch 438 | training: loss: 0.5387, acc: 0.8171, auc:  0.7981 | testing: loss: 0.5340, acc: 0.8266, auc:  0.8023 | time: 74.61s, average batch time: 18.65s
epoch 439 | training: loss: 0.5441, acc: 0.8210, auc:  0.7936 | testing: loss: 0.5217, acc: 0.8264, auc:  0.7991 | time: 74.06s, average batch time: 18.51s
epoch 440 | training: loss: 0.5430, acc: 0.8184, auc:  0.8000 | testing: loss: 0.5274, acc: 0.8189, auc:  0.7739 | time: 74.19s, average batch time: 18.55s
epoch 435 - 440 | training: loss: 0.5408, acc: 0.8171, auc:  0.7958 | testing: loss: 0.5323, acc: 0.8180, auc:  0.7887
epoch 441 | training: loss: 0.5360, acc: 0.8219, auc:  0.7959 | testing: loss: 0.5416, acc: 0.8201, auc:  0.7982 | time: 74.00s, average batch time: 18.50s
epoch 442 | training: loss: 0.5400, acc: 0.8192, auc:  0.7964 | testing: loss: 0.5284, acc: 0.8219, auc:  0.7933 | time: 73.82s, average batch time: 18.45s
epoch 443 | training: loss: 0.5405, acc: 0.8203, auc:  0.7972 | testing: loss: 0.5327, acc: 0.8058, auc:  0.7769 | time: 74.35s, average batch time: 18.59s
epoch 444 | training: loss: 0.5402, acc: 0.8218, auc:  0.8000 | testing: loss: 0.5195, acc: 0.8383, auc:  0.8146 | time: 74.39s, average batch time: 18.60s
epoch 445 | training: loss: 0.5435, acc: 0.8185, auc:  0.7960 | testing: loss: 0.5412, acc: 0.8196, auc:  0.7829 | time: 74.86s, average batch time: 18.72s
epoch 440 - 445 | training: loss: 0.5401, acc: 0.8203, auc:  0.7971 | testing: loss: 0.5327, acc: 0.8212, auc:  0.7932
epoch 446 | training: loss: 0.5385, acc: 0.8215, auc:  0.8010 | testing: loss: 0.5330, acc: 0.8185, auc:  0.7923 | time: 73.94s, average batch time: 18.48s
epoch 447 | training: loss: 0.5471, acc: 0.8164, auc:  0.7918 | testing: loss: 0.5183, acc: 0.8394, auc:  0.8020 | time: 74.20s, average batch time: 18.55s
epoch 448 | training: loss: 0.5417, acc: 0.8172, auc:  0.7946 | testing: loss: 0.5401, acc: 0.8096, auc:  0.7872 | time: 74.03s, average batch time: 18.51s
epoch 449 | training: loss: 0.5462, acc: 0.8137, auc:  0.7888 | testing: loss: 0.5216, acc: 0.8215, auc:  0.7928 | time: 74.54s, average batch time: 18.63s
epoch 450 | training: loss: 0.5366, acc: 0.8213, auc:  0.7943 | testing: loss: 0.5502, acc: 0.8024, auc:  0.7879 | time: 74.45s, average batch time: 18.61s
epoch 445 - 450 | training: loss: 0.5420, acc: 0.8180, auc:  0.7941 | testing: loss: 0.5326, acc: 0.8183, auc:  0.7924
epoch 451 | training: loss: 0.5389, acc: 0.8169, auc:  0.8029 | testing: loss: 0.5279, acc: 0.8272, auc:  0.8028 | time: 73.67s, average batch time: 18.42s
epoch 452 | training: loss: 0.5344, acc: 0.8267, auc:  0.8058 | testing: loss: 0.5509, acc: 0.7959, auc:  0.7636 | time: 74.15s, average batch time: 18.54s
epoch 453 | training: loss: 0.5427, acc: 0.8178, auc:  0.7959 | testing: loss: 0.5159, acc: 0.8359, auc:  0.7953 | time: 74.57s, average batch time: 18.64s
epoch 454 | training: loss: 0.5428, acc: 0.8209, auc:  0.8002 | testing: loss: 0.5266, acc: 0.8103, auc:  0.7743 | time: 74.14s, average batch time: 18.53s
epoch 455 | training: loss: 0.5399, acc: 0.8182, auc:  0.7879 | testing: loss: 0.5353, acc: 0.8292, auc:  0.8192 | time: 74.45s, average batch time: 18.61s
epoch 450 - 455 | training: loss: 0.5397, acc: 0.8201, auc:  0.7986 | testing: loss: 0.5313, acc: 0.8197, auc:  0.7911
epoch 456 | training: loss: 0.5350, acc: 0.8215, auc:  0.8005 | testing: loss: 0.5505, acc: 0.8113, auc:  0.7896 | time: 73.62s, average batch time: 18.41s
epoch 457 | training: loss: 0.5437, acc: 0.8217, auc:  0.8019 | testing: loss: 0.5330, acc: 0.8053, auc:  0.7747 | time: 74.63s, average batch time: 18.66s
epoch 458 | training: loss: 0.5371, acc: 0.8167, auc:  0.7901 | testing: loss: 0.5390, acc: 0.8342, auc:  0.8097 | time: 73.95s, average batch time: 18.49s
epoch 459 | training: loss: 0.5489, acc: 0.8095, auc:  0.7849 | testing: loss: 0.5254, acc: 0.8168, auc:  0.7913 | time: 73.91s, average batch time: 18.48s
epoch 460 | training: loss: 0.5454, acc: 0.8149, auc:  0.7903 | testing: loss: 0.5178, acc: 0.8280, auc:  0.8098 | time: 74.57s, average batch time: 18.64s
epoch 455 - 460 | training: loss: 0.5420, acc: 0.8169, auc:  0.7935 | testing: loss: 0.5331, acc: 0.8191, auc:  0.7950
epoch 461 | training: loss: 0.5391, acc: 0.8176, auc:  0.7952 | testing: loss: 0.5348, acc: 0.8449, auc:  0.8195 | time: 74.22s, average batch time: 18.56s
epoch 462 | training: loss: 0.5420, acc: 0.8214, auc:  0.8010 | testing: loss: 0.5233, acc: 0.8147, auc:  0.7865 | time: 73.84s, average batch time: 18.46s
epoch 463 | training: loss: 0.5416, acc: 0.8157, auc:  0.7957 | testing: loss: 0.5286, acc: 0.8298, auc:  0.7947 | time: 74.31s, average batch time: 18.58s
epoch 464 | training: loss: 0.5385, acc: 0.8240, auc:  0.8026 | testing: loss: 0.5366, acc: 0.8047, auc:  0.7895 | time: 74.20s, average batch time: 18.55s
epoch 465 | training: loss: 0.5414, acc: 0.8122, auc:  0.7951 | testing: loss: 0.5290, acc: 0.8135, auc:  0.7819 | time: 73.87s, average batch time: 18.47s
epoch 460 - 465 | training: loss: 0.5405, acc: 0.8182, auc:  0.7979 | testing: loss: 0.5305, acc: 0.8215, auc:  0.7944
epoch 466 | training: loss: 0.5419, acc: 0.8185, auc:  0.7943 | testing: loss: 0.5285, acc: 0.8237, auc:  0.8026 | time: 74.35s, average batch time: 18.59s
epoch 467 | training: loss: 0.5405, acc: 0.8161, auc:  0.7965 | testing: loss: 0.5214, acc: 0.8281, auc:  0.8071 | time: 74.73s, average batch time: 18.68s
epoch 468 | training: loss: 0.5463, acc: 0.8106, auc:  0.7920 | testing: loss: 0.5283, acc: 0.8325, auc:  0.8019 | time: 74.16s, average batch time: 18.54s
epoch 469 | training: loss: 0.5371, acc: 0.8208, auc:  0.7989 | testing: loss: 0.5462, acc: 0.8034, auc:  0.7771 | time: 75.44s, average batch time: 18.86s
epoch 470 | training: loss: 0.5427, acc: 0.8171, auc:  0.7968 | testing: loss: 0.5309, acc: 0.8181, auc:  0.7835 | time: 74.38s, average batch time: 18.59s
epoch 465 - 470 | training: loss: 0.5417, acc: 0.8166, auc:  0.7957 | testing: loss: 0.5311, acc: 0.8212, auc:  0.7944
epoch 471 | training: loss: 0.5400, acc: 0.8266, auc:  0.8049 | testing: loss: 0.5388, acc: 0.8074, auc:  0.7743 | time: 74.22s, average batch time: 18.55s
epoch 472 | training: loss: 0.5391, acc: 0.8226, auc:  0.7994 | testing: loss: 0.5219, acc: 0.8204, auc:  0.7965 | time: 73.68s, average batch time: 18.42s
epoch 473 | training: loss: 0.5373, acc: 0.8231, auc:  0.8000 | testing: loss: 0.5469, acc: 0.8133, auc:  0.7887 | time: 74.31s, average batch time: 18.58s
epoch 474 | training: loss: 0.5399, acc: 0.8178, auc:  0.8008 | testing: loss: 0.5280, acc: 0.8343, auc:  0.8069 | time: 74.36s, average batch time: 18.59s
epoch 475 | training: loss: 0.5390, acc: 0.8163, auc:  0.7974 | testing: loss: 0.5261, acc: 0.8246, auc:  0.8026 | time: 74.03s, average batch time: 18.51s
epoch 470 - 475 | training: loss: 0.5390, acc: 0.8213, auc:  0.8005 | testing: loss: 0.5324, acc: 0.8200, auc:  0.7938
epoch 476 | training: loss: 0.5405, acc: 0.8168, auc:  0.7978 | testing: loss: 0.5399, acc: 0.8152, auc:  0.7948 | time: 74.89s, average batch time: 18.72s
epoch 477 | training: loss: 0.5430, acc: 0.8179, auc:  0.7956 | testing: loss: 0.5397, acc: 0.8108, auc:  0.7867 | time: 74.29s, average batch time: 18.57s
epoch 478 | training: loss: 0.5465, acc: 0.8207, auc:  0.8032 | testing: loss: 0.5193, acc: 0.8206, auc:  0.7710 | time: 74.17s, average batch time: 18.54s
epoch 479 | training: loss: 0.5374, acc: 0.8161, auc:  0.7952 | testing: loss: 0.5406, acc: 0.8222, auc:  0.8060 | time: 76.56s, average batch time: 19.14s
epoch 480 | training: loss: 0.5408, acc: 0.8159, auc:  0.7930 | testing: loss: 0.5331, acc: 0.8158, auc:  0.7906 | time: 83.61s, average batch time: 20.90s
epoch 475 - 480 | training: loss: 0.5416, acc: 0.8175, auc:  0.7970 | testing: loss: 0.5345, acc: 0.8169, auc:  0.7899
epoch 481 | training: loss: 0.5432, acc: 0.8164, auc:  0.7998 | testing: loss: 0.5178, acc: 0.8317, auc:  0.8046 | time: 84.33s, average batch time: 21.08s
epoch 482 | training: loss: 0.5392, acc: 0.8204, auc:  0.8008 | testing: loss: 0.5262, acc: 0.8235, auc:  0.7937 | time: 80.10s, average batch time: 20.03s
epoch 483 | training: loss: 0.5399, acc: 0.8166, auc:  0.7909 | testing: loss: 0.5246, acc: 0.8330, auc:  0.8055 | time: 79.14s, average batch time: 19.78s
epoch 484 | training: loss: 0.5364, acc: 0.8258, auc:  0.8083 | testing: loss: 0.5333, acc: 0.8133, auc:  0.7934 | time: 78.11s, average batch time: 19.53s
epoch 485 | training: loss: 0.5379, acc: 0.8228, auc:  0.8002 | testing: loss: 0.5475, acc: 0.8119, auc:  0.7893 | time: 87.21s, average batch time: 21.80s
epoch 480 - 485 | training: loss: 0.5393, acc: 0.8204, auc:  0.8000 | testing: loss: 0.5299, acc: 0.8227, auc:  0.7973
epoch 486 | training: loss: 0.5403, acc: 0.8165, auc:  0.8006 | testing: loss: 0.5377, acc: 0.8090, auc:  0.7767 | time: 85.83s, average batch time: 21.46s
epoch 487 | training: loss: 0.5448, acc: 0.8168, auc:  0.7983 | testing: loss: 0.5215, acc: 0.8119, auc:  0.7955 | time: 78.12s, average batch time: 19.53s
epoch 488 | training: loss: 0.5375, acc: 0.8171, auc:  0.7999 | testing: loss: 0.5419, acc: 0.8200, auc:  0.8019 | time: 77.94s, average batch time: 19.49s
epoch 489 | training: loss: 0.5395, acc: 0.8188, auc:  0.8001 | testing: loss: 0.5263, acc: 0.8313, auc:  0.8159 | time: 78.02s, average batch time: 19.50s
epoch 490 | training: loss: 0.5409, acc: 0.8180, auc:  0.7952 | testing: loss: 0.5410, acc: 0.8155, auc:  0.7931 | time: 78.50s, average batch time: 19.63s
epoch 485 - 490 | training: loss: 0.5406, acc: 0.8174, auc:  0.7988 | testing: loss: 0.5337, acc: 0.8176, auc:  0.7966
epoch 491 | training: loss: 0.5392, acc: 0.8200, auc:  0.7956 | testing: loss: 0.5426, acc: 0.8086, auc:  0.7882 | time: 82.53s, average batch time: 20.63s
epoch 492 | training: loss: 0.5414, acc: 0.8194, auc:  0.8019 | testing: loss: 0.5238, acc: 0.8180, auc:  0.7807 | time: 78.42s, average batch time: 19.61s
epoch 493 | training: loss: 0.5426, acc: 0.8205, auc:  0.8001 | testing: loss: 0.5314, acc: 0.8082, auc:  0.7600 | time: 78.24s, average batch time: 19.56s
epoch 494 | training: loss: 0.5375, acc: 0.8225, auc:  0.7995 | testing: loss: 0.5349, acc: 0.8168, auc:  0.7884 | time: 79.00s, average batch time: 19.75s
epoch 495 | training: loss: 0.5374, acc: 0.8183, auc:  0.7952 | testing: loss: 0.5437, acc: 0.8202, auc:  0.7983 | time: 80.33s, average batch time: 20.08s
epoch 490 - 495 | training: loss: 0.5396, acc: 0.8201, auc:  0.7985 | testing: loss: 0.5353, acc: 0.8144, auc:  0.7831
epoch 496 | training: loss: 0.5358, acc: 0.8184, auc:  0.7973 | testing: loss: 0.5438, acc: 0.8286, auc:  0.7985 | time: 80.67s, average batch time: 20.17s
epoch 497 | training: loss: 0.5455, acc: 0.8156, auc:  0.7974 | testing: loss: 0.5191, acc: 0.8293, auc:  0.7931 | time: 80.30s, average batch time: 20.08s
epoch 498 | training: loss: 0.5411, acc: 0.8150, auc:  0.7900 | testing: loss: 0.5370, acc: 0.8202, auc:  0.7952 | time: 78.77s, average batch time: 19.69s
epoch 499 | training: loss: 0.5401, acc: 0.8222, auc:  0.7996 | testing: loss: 0.5301, acc: 0.8112, auc:  0.7886 | time: 83.28s, average batch time: 20.82s
epoch 500 | training: loss: 0.5401, acc: 0.8163, auc:  0.7942 | testing: loss: 0.5468, acc: 0.8062, auc:  0.7862 | time: 78.91s, average batch time: 19.73s
epoch 495 - 500 | training: loss: 0.5405, acc: 0.8175, auc:  0.7957 | testing: loss: 0.5354, acc: 0.8191, auc:  0.7923
"""

# Function to calculate average metrics for specified epoch range


# Function to calculate average metrics for specified epoch range


def parse_auc_from_line(line):
    # 正则表达式匹配auc: 开头的数字（包括小数点）
    match = re.search(r'auc:\s+(\d+\.\d+)', line)
    if match:
        return float(match.group(1))
    else:
        return None


def calculate_average_auc(data_lines, start_epoch, end_epoch):
    auc_values = []
    for line in data_lines:
        if line.startswith(f'epoch {start_epoch}') or line.startswith(f'epoch {end_epoch}'):
            auc = parse_auc_from_line(line)
            if auc is not None:
                auc_values.append(auc)
    if auc_values:
        return sum(auc_values) / len(auc_values)
    else:
        return None


lines = data_str.strip().split('\n')

start_epoch = 1
end_epoch = 500
step = 5
total_average_auc = 0.0
count = 0

for i in range(start_epoch, end_epoch+1, step):
    epoch_start = i
    epoch_end = min(i+step-1, end_epoch)
    if epoch_start <= epoch_end:
        average_auc = calculate_average_auc(lines, epoch_start, epoch_end)
        if average_auc is not None:
            print(
                f'Average AUC for epoch {epoch_start}-{epoch_end}: {average_auc}')
            total_average_auc += average_auc
            count += 1

if count > 0:
    final_average_auc = total_average_auc / count
    print(f'\nFinal Average AUC across all epochs: {final_average_auc}')
else:
    print('No valid AUC values found.')
