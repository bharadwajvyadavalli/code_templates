
def is_symmetric(root):
    """
    Check if a binary tree is symmetric
    """
    # helper function to check if two nodes are symmetric
    def is_mirror(node1, node2):
        # if both nodes are None, they are symmetric
        if not node1 and not node2:
            return True
        # if one of the nodes is None, they are not symmetric
        if not node1 or not node2:
            return False
        # if the values of the nodes are not equal, they are not symmetric
        if node1.val != node2.val:
            return False
        # check if the left subtree of the first node is symmetric to the right subtree of the second node
        left = is_mirror(node1.left, node2.right)
        # check if the right subtree of the first node is symmetric to the left subtree of the second node
        right = is_mirror(node1.right, node2.left)
        # return True if both subtrees are symmetric, False otherwise
        return left and right
    # check if the root is symmetric to itself
    return is_mirror(root, root)

def max_depth(root):
    """
    Find the maximum depth of a binary tree
    """
    # if the root is None, the depth is 0
    if not root:
        return 0
    # calculate the depth of the left subtree
    left = max_depth(root.left)
    # calculate the depth of the right subtree
    right = max_depth(root.right)
    # return the maximum depth of the two subtrees plus 1
    return max(left, right) + 1

def is_balanced(root):
    """
    Check if a binary tree is balanced
    """
    # helper function to check if a subtree is balanced
    def is_balanced_helper(node):
        # if the node is None, the subtree is balanced
        if not node:
            return 0
        # calculate the height of the left subtree
        left = is_balanced_helper(node.left)
        # calculate the height of the right subtree
        right = is_balanced_helper(node.right)
        # if the subtree is not balanced, return -1
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1
        # return the height of the subtree
        return max(left, right) + 1
    # check if the root is balanced
    return is_balanced_helper(root) != -1

def level_order(root):
    """
    Traverse a binary tree in level order
    """
    # if the root is None, return an empty list
    if not root:
        return []
    # create a queue to store the nodes
    queue = [root]
    # create a list to store the level order traversal
    result = []
    # iterate through the nodes in the queue
    while queue:
        # create a list to store the nodes at the current level
        level = []
        # iterate through the nodes in the current level
        for _ in range(len(queue)):
            # pop the first node from the queue
            node = queue.pop(0)
            # add the value of the node to the current level
            level.append(node.val)
            # add the left child of the node to the queue
            if node.left:
                queue.append(node.left)
            # add the right child of the node to the queue
            if node.right:
                queue.append(node.right)
        # add the current level to the level order traversal
        result.append(level)
    # return the level order traversal
    return result

def zigzag_level_order(root):
    """
    Traverse a binary tree in zigzag level order
    """
    # if the root is None, return an empty list
    if not root:
        return []
    # create a queue to store the nodes
    queue = [root]
    # create a list to store the zigzag level order traversal
    result = []
    # set the flag to indicate the direction of traversal
    flag = 1
    # iterate through the nodes in the queue
    while queue:
        # create a list to store the nodes at the current level
        level = []
        # iterate through the nodes in the current level
        for _ in range(len(queue)):
            # pop the first node from the queue
            node = queue.pop(0)
            # add the value of the node to the current level
            level.append(node.val)
            # add the left child of the node to the queue
            if node.left:
                queue.append(node.left)
            # add the right child of the node to the queue
            if node.right:
                queue.append(node.right)
        # add the current level to the zigzag level order traversal
        result.append(level[::flag])
        # reverse the direction of traversal
        flag *= -1
    # return the zigzag level order traversal
    return result

def invert_tree(root):
    """
    Invert a binary tree
    """
    # if the root is None, return None
    if not root:
        return None
    # invert the left subtree
    left = invert_tree(root.left)
    # invert the right subtree
    right = invert_tree(root.right)
    # swap the left and right subtrees
    root.left = right
    root.right = left
    # return the inverted tree
    return root

def lowest_common_ancestor(root, p, q):
    """
    Find the lowest common ancestor of two nodes in a binary tree
    """
    # if the root is None or one of the nodes is the root, return the root
    if not root or root == p or root == q:
        return root
    # find the lowest common ancestor in the left subtree
    left = lowest_common_ancestor(root.left, p, q)
    # find the lowest common ancestor in the right subtree
    right = lowest_common_ancestor(root.right, p, q)
    # if both nodes are found in different subtrees, return the root
    if left and right:
        return root
    # if one of the nodes is found, return the node
    return left or right

def path_sum(root, target):
    """
    Find all root-to-leaf paths that sum up to a given target
    """
    # helper function to find all root-to-leaf paths
    def path_sum_helper(node, target, path, result):
        # if the node is None, return
        if not node:
            return
        # add the value of the node to the path
        path.append(node.val)
        # if the node is a leaf and the path sums up to the target, add the path to the result
        if not node.left and not node.right and sum(path) == target:
            result.append(list(path))
        # find all root-to-leaf paths in the left subtree
        path_sum_helper(node.left, target, path, result)
        # find all root-to-leaf paths in the right subtree
        path_sum_helper(node.right, target, path, result)
        # remove the value of the node from the path
        path.pop()
    # create a list to store the root-to-leaf paths
    result = []
    # find all root-to-leaf paths that sum up to the target
    path_sum_helper(root, target, [], result)
    # return the result
    return result

def is_same_tree(p, q):
    """
    Check if two binary trees are the same
    """
    # if both nodes are None, they are the same
    if not p and not q:
        return True
    # if one of the nodes is None, they are not the same
    if not p or not q:
        return False
    # if the values of the nodes are not equal, they are not the same
    if p.val != q.val:
        return False
    # check if the left subtrees are the same
    left = is_same_tree(p.left, q.left)
    # check if the right subtrees are the same
    right = is_same_tree(p.right, q.right)
    # return True if both subtrees are the same, False otherwise
    return left and right

def left_view(root):
    """
    Find the left view of a binary tree
    """
    # if the root is None, return an empty list
    if not root:
        return []
    # create a queue to store the nodes
    queue = [root]
    # create a list to store the left view of the tree
    result = []
    # iterate through the nodes in the queue
    while queue:
        # add the value of the first node to the left view
        result.append(queue[0].val)
        # create a list to store the nodes at the current level
        level = []
        # iterate through the nodes in the current level
        for node in queue:
            # add the left child of the node to the queue
            if node.left:
                level.append(node.left)
            # add the right child of the node to the queue
            if node.right:
                level.append(node.right)
        # update the queue with the nodes at the next level
        queue = level
    # return the left view of the tree
    return result

def right_view(root):
    """
    Find the right view of a binary tree
    """
    # if the root is None, return an empty list
    if not root:
        return []
    # create a queue to store the nodes
    queue = [root]
    # create a list to store the right view of the tree
    result = []
    # iterate through the nodes in the queue
    while queue:
        # add the value of the last node to the right view
        result.append(queue[-1].val)
        # create a list to store the nodes at the current level
        level = []
        # iterate through the nodes in the current level
        for node in queue:
            # add the left child of the node to the queue
            if node.left:
                level.append(node.left)
            # add the right child of the node to the queue
            if node.right:
                level.append(node.right)
        # update the queue with the nodes at the next level
        queue = level
    # return the right view of the tree
    return result

def vertical_order(root):
    """
    Traverse a binary tree in vertical order
    """
    # if the root is None, return an empty list
    if not root:
        return []
    # create a dictionary to store the nodes at each vertical level
    vertical = {}
    # create a queue to store the nodes
    queue = [(root, 0)]
    # iterate through the nodes in the queue
    while queue:
        # pop the first node from the queue
        node, level = queue.pop(0)
        # add the node to the dictionary at the vertical level
        if level in vertical:
            vertical[level].append(node.val)
        else:
            vertical[level] = [node.val]
        # add the left child of the node to the queue
        if node.left:
            queue.append((node.left, level - 1))
        # add the right child of the node to the queue
        if node.right:
            queue.append((node.right, level + 1))
    # sort the nodes at each vertical level
    result = [vertical[key] for key in sorted(vertical)]
    # return the vertical order traversal
    return result

def boundary_of_binary_tree(root):
    """
    Find the boundary of a binary tree
    """
    # if the root is None, return an empty list
    if not root:
        return []
    # create a list to store the boundary of the tree
    boundary = []
    # add the root to the boundary
    boundary.append(root.val)
    # add the left boundary of the tree
    left_boundary(root.left, boundary)
    # add the leaves of the tree
    leaves(root, boundary)
    # add the right boundary of the tree
    right_boundary(root.right, boundary)
    # return the boundary of the tree
    return boundary