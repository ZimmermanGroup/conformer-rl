"""
Storage
=======

Storage utility class for agents.

"""

import torch

class Storage:
    """Saves and stores experiences from the agent.

    Parameters
    ----------
    rollout : int
        The rollout length of the agent in each sampling iteration (per worker).
    workers : int
        The number of workers used for sampling by the agent.

    """
    def __init__(self, rollout: int, workers: int):
        self.storage = {}
        self.rollout = rollout
        self.workers = workers

    def __getitem__(self, key: str) -> list:
        """Returns all batched items in storage with the given `key`.

        :meta public:

        Parameters
        ----------
        key : str
            The key of the items to be accessed.

        Returns
        -------
        list
            A list of all the items with the associated `key` 
            previously appended to the storage.


        """
        return self.storage[key]

    def append(self, data:dict) -> None:
        """Appends item(s) to the storage

        Parameters
        ----------
        data : dict
            The items to be appended.
        """
        for key, val in data.items():
            self.storage.setdefault(key, []).append(val)

    def order(self, key:str) -> list:
        """Splits each batch of items associated with `key` by worker, and then
        orders the items into a list sorted firstly by worker and secondly by the
        order the items were appended to storage.

        Parameters
        ----------
        key: str
            The key associated with the item to be retrieved.

        Returns
        -------
        list
            The list of the items, split by batch and ordered.

        Notes
        -----
        An example of how the ordering is performed: Suppose the storage is initialized
        with `rollout=3` and `workers=2`. Then in each sampling iteration of the agent,
        a batch of 2 items (one for each worker) will be appended to storage for a total
        of 3 iterations. Thus, at the end of the iterations, the storage value for any given item
        will look similar to this::

            [[worker 1 sample 1, worker 2 sample 1],
             [worker 1 sample 2, worker 2 sample 2],
             [worker 1 sample 3, worker 2 sample 3]].

        The list returned by :meth:`~Storage.order` would then be::

            [worker 1 sample 1, worker 1 sample 2, worker 1 sample 3,
            worker 2 sample 1, worker 2 sample 2, worker 3 sample 3].

        The purpose of this ordering is to maintain chronologically consecutive blocks of rollout samples,
        which is useful for agents with recursive components.
        """
        if torch.is_tensor(self.storage[key][0]):
            ordered = torch.stack(self.storage[key][:self.rollout], -2)
            shape = ordered.shape
            ordered = ordered.view(*shape[:-3], self.rollout * self.workers, shape[-1])
            return ordered
        else:
            ordered = []
            for i in range(self.workers):
                ordered += [self.storage[key][j][i] for j in range(self.rollout)]
            return ordered


    def reset(self) -> None:
        """Empties the storage.

        """
        self.storage = {}