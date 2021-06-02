import torch
from conformer_rl.agents.storage import Storage

class Obj:
    def __init__(self, worker, batch):
        self.worker = worker
        self.batch = batch


def test_storage():
    rollout = 5
    workers = 7
    storage = Storage(rollout, workers)
    tensors = torch.rand((rollout, workers, 2))
    for i in range(rollout):
        objects = [Obj(j, i) for j in range(workers)]
        storage.append({
            'objects': objects,
            'tensors': tensors[i]
        })

    storage_objects = storage['objects']
    storage_tensors = storage['tensors']
    assert(len(storage_objects) == rollout)
    assert(len(storage_tensors) == rollout)

    for i in range(rollout):
        for j in range(workers):
            assert(storage_objects[i][j].worker == j)
            assert(storage_objects[i][j].batch == i)


    for i in range(rollout):
        assert(torch.all(torch.eq(storage_tensors[i], tensors[i])))
    
    storage_objects = storage.order('objects')
    storage_tensors = storage.order('tensors')

    for i in range(workers * rollout):
        assert(storage_objects[i].batch == (i % rollout))
        assert(storage_objects[i].worker == (i // rollout))

        assert(torch.all(torch.eq(storage_tensors[i], tensors[i % rollout, i//rollout])))

