import torch
import multiprocessing as mp

x = torch.Tensor(4)
x.cuda()

print(x)


def f():
    print('hmmm')
    y = torch.Tensor(10)
    y.cuda()
    print(y)

p = mp.Process(target=f)
p.start()
p.join()
print('done')


# seperate process doesn't immediately re initialize cuda
# seperate process with a PYTORCH TENSOR doesn't re initialize cuda
# seperate process with a cuda tensor does reinitialize cuda and dies with a message: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
