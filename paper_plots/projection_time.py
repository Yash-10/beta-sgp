# If LaTeX is not installed, run `!sudo apt install cm-super dvipng texlive-latex-extra texlive-latex-recommended`.

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

x=[25**2, 50**2, 100**2, 200**2]
y=[0.03523226, 0.04267109, 0.09127162, 0.24407341]

fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.plot(x, y, c='black', linestyle='--', marker='o')
labels = [r'$25^2$', r'$50^2$', r'$100^2$', r'$200^2$']
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Time (sec)')
ax.set_xlabel('Problem size')
plt.savefig('projection_time.png', dpi=500)
plt.show()
