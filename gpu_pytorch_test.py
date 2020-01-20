import torch
import multiprocessing as mp

x = torch.Tensor(4)
x.cuda()

print(x)


def f():
    print('hmmm')
    y = torch.Tensor(10)
    print(y)

p = mp.Process(target=f)
p.start()
p.join()
print('done')
