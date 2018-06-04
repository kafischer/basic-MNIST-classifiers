from pylab import *
from matplotlib.colors import LogNorm

confusion1 = loadtxt('../training_data/test_confusion_part2.1.txt')
confusion2 = loadtxt('../training_data/test_confusion_part2.2.txt')
confusion3 = loadtxt('../training_data/test_confusion_svm.txt')

confusion1 = confusion1/np.sum(confusion1)
confusion2 = confusion2/np.sum(confusion2)
confusion3 = confusion3/np.sum(confusion3)

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['times']
rcParams['font.size'] = 7
f = figure(figsize=(5.55, 1.7))
f.patch.set_facecolor('white')

subplot(131)
pcolor(confusion1.T, cmap='binary',
       norm=LogNorm(vmin=confusion1.min()+1e-3, vmax=confusion1.max()))
xlabel('Prediction')
ylabel('Label')
colorbar()
xticks(range(10))
yticks(range(10))
gca().annotate('(a)', xy=(0.016, 0.95),
               xycoords='figure fraction', fontsize='8')

subplot(132)
pcolor(confusion2.T, cmap='binary',
       norm=LogNorm(vmin=confusion1.min()+1e-3, vmax=confusion1.max()))
xlabel('Prediction')
ylabel('Label')
colorbar()
xticks(range(10))
yticks(range(10))
gca().annotate('(b)', xy=(0.34, 0.95),
               xycoords='figure fraction', fontsize='8')

subplot(133)
pcolor(confusion3.T, cmap='binary',
       norm=LogNorm(vmin=confusion1.min()+1e-3, vmax=confusion1.max()))
colorbar()
xlabel('Prediction')
ylabel('Label')
xticks(range(10))
yticks(range(10))
gca().annotate('(c)', xy=(0.67, 0.95),
               xycoords='figure fraction', fontsize='8')

tight_layout()
savefig('Fig3.svg')