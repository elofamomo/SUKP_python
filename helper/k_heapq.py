import heapq
import itertools
class TopKHeap:
    def __init__(self, k):
        self.k = k
        self.heap = []  # Min-heap to store the top K largest values
        self.counter = itertools.count()
    def add(self, value, sol):
        count = next(self.counter)
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (value, count, sol))
        elif value > self.heap[0][0]:  # If new value is larger than the smallest in heap
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, (value, count, sol))
    
    def get_top_k(self, sorted_descending=True):
        items = [(v, s) for v, c, s in self.heap]  # Copy to avoid modifying heap
        if sorted_descending:
            return sorted(items, key=lambda x: x[0], reverse=True)  # Largest first, as list of (value, state)
        return sorted(items, key=lambda x: x[0])
    
    def get_top_k_states(self, sorted_by_value_descending=True):
        sorted_items = sorted(self.heap, key=lambda x: x[0], reverse=sorted_by_value_descending)
        return [s for v, c, s in sorted_items]  # Return only states, in the sorted order

    def get_top_k_values(self, sorted_descending=True):
        sorted_items = sorted(self.heap, key=lambda x: x[0], reverse=sorted_descending)
        return [v for v, c, s in sorted_items]  # Return only values, in the sorted order