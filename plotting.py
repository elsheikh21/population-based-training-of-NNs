import matplotlib.pyplot as plt


def plot(best_accuracy_hist, nonreg_accuracy_hist, l1_scale_hist):
       '''
       Plots the unregularized and regularized models' accuracy,
       as well as the l1 regularizer 
       '''
       
       f = plt.figure(figsize=(10, 5))
       ax = f.add_subplot(1, 1, 1)
       ax.plot(best_accuracy_hist)
       ax.plot(nonreg_accuracy_hist, c='red')
       ax.set(xlabel='Hundreds of training iterations', ylabel='Test accuracy')
       ax.legend(['Population based training', 'Non-regularized baseline'])
       plt.savefig('Non-Regularized Baseline model Vs PBT model.png')
       plt.show()

       f = plt.figure(figsize=(10, 5))
       ax = f.add_subplot(1, 1, 1)
       ax.plot(2 ** l1_scale_hist.T)
       ax.set_yscale('log')
       ax.set(xlabel='Hundreds of training iterations',
              ylabel='L1 regularizer scale')
       plt.savefig('L1 Regularized Scale.png')
       plt.show()
