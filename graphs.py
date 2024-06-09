
def is_bipartite(graph):
    """
    Check if a graph is bipartite
    """
    # create a dictionary to store the colors of the nodes
    colors = {}
    # iterate through the nodes in the graph
    for node in graph:
        # if the node is not colored, color it and add it to the queue
        if node not in colors:
            colors[node] = 1
            queue = [node]
            # iterate through the queue
            while queue:
                current = queue.pop(0)
                # iterate through the neighbors of the current node
                for neighbor in graph[current]:
                    # if the neighbor is not colored, color it and add it to the queue
                    if neighbor not in colors:
                        colors[neighbor] = 1 - colors[current]
                        queue.append(neighbor)
                    # if the neighbor has the same color as the current node, return False
                    elif colors[neighbor] == colors[current]:
                        return False
    # if no conflicts are found, return True
    return True

def is_connected(graph):
    """
    Check if a graph is connected
    """
    # create a set to store the visited nodes
    visited = set()
    # create a queue to store the nodes to visit
    queue = [next(iter(graph))]
    # iterate through the queue
    while queue:
        current = queue.pop(0)
        # add the current node to the visited set
        visited.add(current)
        # iterate through the neighbors of the current node
        for neighbor in graph[current]:
            # if the neighbor has not been visited, add it to the queue
            if neighbor not in visited:
                queue.append(neighbor)
    # if all nodes are visited, return True
    return len(visited) == len(graph)

def is_cyclic(graph):
    """
    Check if a graph contains a cycle
    """
    # create a set to store the visited nodes
    visited = set()
    # create a set to store the nodes in the current path
    path = set()
    # iterate through the nodes in the graph
    for node in graph:
        # if the node has not been visited, check for a cycle
        if node not in visited:
            if is_cyclic_helper(graph, node, visited, path):
                return True
    # if no cycle is found, return False
    return False

def is_cyclic_helper(graph, node, visited, path):
    """
    Helper function to check for a cycle in a graph
    """
    # add the current node to the visited and path sets
    visited.add(node)
    path.add(node)
    # iterate through the neighbors of the current node
    for neighbor in graph[node]:
        # if the neighbor is in the path set, a cycle is found
        if neighbor in path:
            return True
        # if the neighbor has not been visited, check for a cycle
        if neighbor not in visited:
            if is_cyclic_helper(graph, neighbor, visited, path):
                return True
    # remove the current node from the path set
    path.remove(node)
    # if no cycle is found, return False
    return False

def topological_sort(graph):
    """
    Perform a topological sort on a graph
    """
    # create a dictionary to store the indegree of each node
    indegree = {node: 0 for node in graph}
    # iterate through the nodes in the graph
    for node in graph:
        # iterate through the neighbors of the current node
        for neighbor in graph[node]:
            # increment the indegree of the neighbor
            indegree[neighbor] += 1
    # create a queue to store the nodes with an indegree of 0
    queue = [node for node in indegree if indegree[node] == 0]
    # create a list to store the sorted nodes
    result = []
    # iterate through the queue
    while queue:
        current = queue.pop(0)
        # add the current node to the result list
        result.append(current)
        # iterate through the neighbors of the current node
        for neighbor in graph[current]:
            # decrement the indegree of the neighbor
            indegree[neighbor] -= 1
            # if the indegree of the neighbor is 0, add it to the queue
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    # if all nodes are visited, return the result list
    return result if len(result) == len(graph) else []

def dijkstra(graph, start):
    """
    Perform Dijkstra's algorithm on a graph
    """
    # create a dictionary to store the distances from the start node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    # create a priority queue to store the nodes to visit
    queue = [(0, start)]
    # iterate through the queue
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        # if the current distance is greater than the distance from the start node, skip
        if current_distance > distances[current_node]:
            continue
        # iterate through the neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            # if the distance is less than the current distance, update the distance and add the neighbor to the queue
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    # return the distances from the start node
    return distances

class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        return True
