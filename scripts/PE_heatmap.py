import matplotlib.pyplot as plt
import torch


max_sen_len = 308
D = 512
pos_encoding = torch.zeros(max_sen_len, D)
for pos in range(max_sen_len):
    for i in range(D):
        exponent = pos / (10000 ** (2 * i / D))
        exponent = torch.FloatTensor([exponent])
        if i % 2 == 0:
            pos_encoding[pos][i] = torch.sin(exponent)
        else:
            pos_encoding[pos][i] = torch.cos(exponent)
plt.imshow(pos_encoding, cmap='viridis')
plt.colorbar()
plt.show()
