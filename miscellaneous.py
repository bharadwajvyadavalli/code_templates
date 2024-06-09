# sample code usage of sets

def add_to_set(s, x):
    """
    Add an element to a set
    """
    s.add(x)
    return s

def remove_from_set(s, x):
    """
    Remove an element from a set
    """
    s.remove(x)
    return s

def check_in_set(s, x):
    """
    Check if an element is in a set
    """
    return x in s

def iterate_over_set(s):
    """
    Iterate over the elements of a set
    """
    for x in s:
        print(x)

def add_to_hashtable(ht, k, v):
    """
    Add a key-value pair to a hashtable
    """
    ht[k] = v
    return ht

def remove_from_hashtable(ht, k):
    """
    Remove a key from a hashtable
    """
    del ht[k]
    return ht

def check_in_hashtable(ht, k):
    """
    Check if a key is in a hashtable
    """
    return k in ht

def iterate_over_hashtable(ht):
    """
    Iterate over the key-value pairs of a hashtable
    """
    for k, v in ht.items():
        print(k, v)

#sample code usage of heaps
def add_to_heap(h, x):
    """
    Add an element to a heap
    """
    heapq.heappush(h, x)
    return h

def remove_from_heap(h):
    """
Remove the smallest element from a heap
"""
    return heapq.heappop(h)

def get_smallest_from_heap(h):
    """
    Get the smallest element from a heap
    """
    return h[0]

def heapify_list(arr):
    """
    Heapify a list
    """
    heapq.heapify(arr)
    return arr

def get_largest_from_heap(h):
    """
    Get the largest element from a heap
    """
    return max(h)

def get_smallest_k_elements(arr, k):
    """
    Get the smallest k elements from a list
    """
    return heapq.nsmallest(k, arr)

def get_largest_k_elements(arr, k):
    """
    Get the largest k elements from a list
    """
    return heapq.nlargest(k, arr)
