import matplotlib.pyplot as plt
import numpy as np


d_model = 512
warmup_steps = 4000
step_num = 0
lr_schedule = []
for _ in range(100000):
    step_num += 1
    lr = d_model**(-0.5) * np.minimum(step_num**(-0.5), step_num*(warmup_steps**(-1.5)))
    lr_schedule.append(lr)

plt.plot(lr_schedule)
plt.xlabel('step')
plt.ylabel('lr')
plt.title('lr schedule')
plt.show()
