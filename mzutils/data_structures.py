import heapq


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
