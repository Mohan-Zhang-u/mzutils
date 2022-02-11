import heapq
import os
import pandas as pd
        
from .os_funcs import mkdir_p, get_things_in_loc


class SimplePriorityQueue():
    """
    a simple wrapper around heapq.
    
    >>> q = SimplePriorityQueue()
    >>> q.put((2, "Harry"))
    >>> q.put((3, "Charles"))
    >>> q.put((1, "Riya"))
    >>> q.put((4, "Stacy"))
    >>> q.put((0, "John"))
    >>> print(q.nlargest(3))
    [(4, 'Stacy'), (3, 'Charles'), (2, 'Harry')]
    >>> print(q.nsmallest(8))
    [(1, 'Riya'), (2, 'Harry'), (3, 'Charles'), (4, 'Stacy')]
    >>> print(q.get())
    (1, 'Riya')
    >>> print(q.get())
    (2, 'Harry')
    >>> print(q.get())
    (3, 'Charles')
    >>> print(q.get())
    (4, 'Stacy')
    >>> print(q.get())
    None
    """

    def __init__(self, maxsize=0):
        self.heap = []
        self.maxsize = maxsize
        heapq.heapify(self.heap)

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return self.heap.__str__()

    def put(self, element):
        # like indicated here, https://stackoverflow.com/questions/42236820/adding-numpy-array-to-a-heap-queue, inserting numpy array to heapq can be risky.
        try:
            numpy_checker = element in self.heap
        except:
            raise ValueError(
                "Exact same value put again into the priority queue! You may trigger numpy comparision error, please check and fix.")
        if self.maxsize > 0 and len(self.heap) >= self.maxsize:
            heapq.heappushpop(self.heap, element)
        else:
            heapq.heappush(self.heap, element)

    def get(self):
        try:
            return heapq.heappop(self.heap)
        except IndexError:
            return None

    def nlargest(self, n, key=None):
        return heapq.nlargest(n, self.heap)

    def nsmallest(self, n, key=None):
        return heapq.nsmallest(n, self.heap)


class SeedData:
    """
    A dictionary that aims to average the evaluated mean_episode_return accross different random seed.
    Also controls where to resume the experiments from.
    """

    def __init__(self, save_path, seeds, resume_from={}):
        self.seeds = seeds
        self.seed_data = pd.DataFrame({
            'algo_name': pd.Series([], dtype='str'),
            'test_reward': pd.Series([], dtype='float'),
            'seed': pd.Series([], dtype='int'),
        })
        self.save_path = save_path
        mkdir_p(save_path)
        self.load()
        # set experiment range
        self.resume_from = resume_from
        self.resume_check_passed = False

    def load(self):
        re_list = get_things_in_loc(self.save_path)
        if not re_list:
            print("Cannot find the a seed_data.csv at", self.save_path, "initializing a new one.")
            self.seed_data.to_csv(os.path.join(self.save_path, 'seed_data.csv'), index=False)
        else:
            self.seed_data = pd.read_csv(os.path.join(self.save_path, 'seed_data.csv'))
            print("Loaded the seed_data.csv at", self.save_path)
    
    def save(self):
        self.seed_data.to_csv(os.path.join(self.save_path, 'seed_data.csv'), index=False)

    def append(self, algo_name, test_reward, seed):
        self.seed_data.loc[len(self.seed_data)] = [algo_name, test_reward, seed]

    def setter(self, algo_name, test_reward, seed):
        # average over seed makes seed==-1
        # online makes dataset_percent==0.0
        self.append(algo_name, test_reward, seed)
        averaged_reward = self.seed_data.loc[(self.seed_data['algo_name'] == algo_name)]['test_reward'].mean()
        if seed == self.seeds[-1]: # append the average, seed now set to -1
            self.seed_data.loc[len(self.seed_data)] = [algo_name, averaged_reward, -1]
        self.save()
        return averaged_reward

    def resume_checker(self, current_positions):
        """
        current_positions has the same shape as self.resume_from
        return True if the current loop still need to be skipped.
        """
        if self.resume_check_passed is True: # checker has already passed.
            return True

        if not self.resume_from:
            self.resume_check_passed = True
        elif all([self.resume_from[condition] is None for condition in self.resume_from]):
            self.resume_check_passed = True
        else:
            self.resume_check_passed = all([self.resume_from[condition] == current_positions[condition] for condition in self.resume_from])
        return self.resume_check_passed
