# If LaTeX is not installed, run `!sudo apt install cm-super dvipng texlive-latex-extra texlive-latex-recommended`.

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True) 
x=[25**2, 50**2, 100**2, 200**2, 400**2, 800**2, 1600**2]
y=[0.0001361819999601721, 0.00023590719997628183, 0.0003930150000087451, 0.001102002000061475, 0.0055610119999983, 0.0470264084000064, 0.2175287432000914] 
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.plot(x, y, c='black', linestyle='-', marker='o', label='FC-SGP projection')
coefficients = np.polyfit(x*np.log(x),y,1) # Use log(x) as the input to polyfit.
fit = np.poly1d(coefficients)
ax.plot(x,fit(x*np.log(x)),"--", label="$O(nlogn)$", c='#1C9099')

#coeffs = np.polyfit(np.array(x)**2, y, 1)
#fitq = np.poly1d(coeffs)
#ax.plot(x, fitq(np.array(x)**2), "--", label="$O(n^2)$", c='#FC9272')

#m, b = np.polyfit(x, y, 1)
#ax.plot(x, m*np.array(x) + b, c='#FC9272')

labels = [r'$25^2$', ' ', ' ', ' ', r'$400^2$', r'$800^2$', r'$1600^2$']
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Time (sec)')
ax.set_xlabel('Problem size')
ax.legend()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.savefig('projection_time.png', bbox_inches='tight', dpi=500)
plt.show()
