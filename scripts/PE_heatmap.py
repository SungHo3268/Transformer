import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm


max_sen_len = 128
D = 1024

pos_encoding = torch.zeros(max_sen_len, D)
for pos in tqdm(range(max_sen_len)):
    for i in range(D):
        exponent = pos / (10000 ** (2 * i / D))
        exponent = torch.FloatTensor([exponent])
        if i % 2 == 0:
            pos_encoding[pos][i] = torch.sin(exponent)
        else:
            pos_encoding[pos][i] = torch.cos(exponent)
pos_encoding = pos_encoding[:, : 512]
plt.imshow(pos_encoding, cmap='viridis')
plt.colorbar()
plt.show()
