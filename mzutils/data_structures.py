import heapq


class SimplePriorityQueue():
    """
    a simple wrapper around heapq.
    
    >>> q = SimplePriorityQueue()
    >>> q.put((2, "Harry"))
    >>> q.put((3, "Charles"))
    >>> q.put((1, "Riya"))
    >>> q.put((4, "Stacy"))
    >>> print(q.nlargest(3))
    [(4, 'Stacy'), (3, 'Charles'), (2, 'Harry')]
    >>> print(q.nsmallest(5))
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
        
    def put(self, element):
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