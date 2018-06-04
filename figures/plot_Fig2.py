from pylab import *

part = '2.1'

test_acc1 = loadtxt('../training_data/test_accuracy_part'+part+'.txt')
train_acc1 = loadtxt('../training_data/train_accuracy_part'+part+'.txt')
test_loss1 = loadtxt('../training_data/test_loss_part'+part+'.txt')
train_loss1 = loadtxt('../training_data/train_loss_part'+part+'.txt')

part = '2.2'

test_acc2 = loadtxt('../training_data/test_accuracy_part'+part+'.txt')
train_acc2 = loadtxt('../training_data/train_accuracy_part'+part+'.txt')
test_loss2 = loadtxt('../training_data/test_loss_part'+part+'.txt')
train_loss2 = loadtxt('../training_data/train_loss_part'+part+'.txt')

svm_acc = loadtxt('../training_data/test_accuracy_svm.txt')

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['times']
rcParams['font.size'] = 7
f = figure(figsize=(5.55, 2.3))
f.patch.set_facecolor('white')

subplot(131)
loglog(train_loss1, 'r', linestyle='dotted')
loglog(train_loss2, 'b', linestyle='dotted')
l1, = loglog(test_loss1, 'r')
l2, = loglog(test_loss2, 'b')
gca().spines['right'].set_visible(False)
gca().spines['top'].set_visible(False)
gca().yaxis.set_ticks_position('left')
gca().xaxis.set_ticks_position('bottom')
xlim(1,200)
xlabel('Epoch (minibatch 200)')
ylabel('Loss')
legend([l1, l2], ['softmax', 'CNN'],
       loc=3, frameon=False, fontsize='7')
gca().annotate('(a)', xy=(0.017, 0.95),
               xycoords='figure fraction', fontsize='8')

subplot(132)
semilogx(train_acc1, 'r', linestyle='dotted')
semilogx(train_acc2, 'b', linestyle='dotted')
l1, = semilogx(test_acc1, 'r')
l2, = semilogx(test_acc2, 'b')
gca().spines['right'].set_visible(False)
gca().spines['top'].set_visible(False)
gca().yaxis.set_ticks_position('left')
gca().xaxis.set_ticks_position('bottom')
xlim(1,200)
xlabel('Epoch (minibatch 200)')
ylabel('Accuracy')
legend([l1, l2], ['softmax', 'CNN'],
       loc=4, frameon=False, fontsize='7')
gca().annotate('(b)', xy=(0.35, 0.95),
               xycoords='figure fraction', fontsize='8')

subplot(133)
semilogx(svm_acc)
semilogx([1, 40], [1.0, 1.0], 'b', linestyle='dotted')
ylim(0.1,1)
xlim(1,200)
gca().spines['right'].set_visible(False)
gca().spines['top'].set_visible(False)
gca().yaxis.set_ticks_position('left')
gca().xaxis.set_ticks_position('bottom')
xlabel('Train set size (200)')
ylabel('Accuracy')
gca().annotate('(c)', xy=(0.676, 0.95),
               xycoords='figure fraction', fontsize='8')

tight_layout()
savefig('Fig2.pdf')