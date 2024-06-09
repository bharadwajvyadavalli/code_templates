
def reverse_linked_list(head):
    """
    Reverse a singly linked list
    """
    # initialize the previous node to None
    prev = None
    # set the current node to the head of the list
    curr = head
    # iterate through the list
    while curr:
        # store the next node
        next_node = curr.next
        # reverse the link
        curr.next = prev
        # move the previous node to the current node
        prev = curr
        # move the current node to the next node
        curr = next_node
    # return the new head of the list
    return prev

def detect_cycle(head):
    """
    Detect a cycle in a linked list
    """
    # initialize two pointers to the head of the list
    slow = head
    fast = head
    # iterate through the list
    while fast and fast.next:
        # move the slow pointer one step
        slow = slow.next
        # move the fast pointer two steps
        fast = fast.next.next
        # if the pointers meet, there is a cycle
        if slow == fast:
            return True
    # if the pointers reach the end of the list, there is no cycle
    return False

def find_cycle_start(head):
    """
    Find the start of a cycle in a linked list
    """
    # initialize two pointers to the head of the list
    slow = head
    fast = head
    # iterate through the list
    while fast and fast.next:
        # move the slow pointer one step
        slow = slow.next
        # move the fast pointer two steps
        fast = fast.next.next
        # if the pointers meet, break out of the loop
        if slow == fast:
            break
    # if the fast pointer reaches the end of the list, there is no cycle
    if not fast or not fast.next:
        return None
    # reset the slow pointer to the head of the list
    slow = head
    # iterate through the list
    while slow != fast:
        # move the slow pointer one step
        slow = slow.next
        # move the fast pointer one step
        fast = fast.next
    # return the start of the cycle
    return slow

def remove_nth_from_end(head, n):
    """
    Remove the nth node from the end of a linked list
    """
    # create a dummy node to handle edge cases
    dummy = ListNode(0)
    dummy.next = head
    # initialize two pointers to the dummy node
    slow = dummy
    fast = dummy
    # move the fast pointer n steps ahead
    for _ in range(n + 1):
        fast = fast.next
    # iterate through the list
    while fast:
        # move the slow pointer one step
        slow = slow.next
        # move the fast pointer one step
        fast = fast.next
    # remove the nth node from the end
    slow.next = slow.next.next
    # return the head of the list
    return dummy.next

def merge_two_lists(l1, l2):
    """
    Merge two sorted linked lists
    """
    # create a dummy node to handle edge cases
    dummy = ListNode(0)
    # set the current node to the dummy node
    curr = dummy
    # iterate through the lists
    while l1 and l2:
        # if the value in the first list is less than the value in the second list
        if l1.val < l2.val:
            # set the next node to the current node
            curr.next = l1
            # move the first list pointer to the next node
            l1 = l1.next
        # otherwise
        else:
            # set the next node to the current node
            curr.next = l2
            # move the second list pointer to the next node
            l2 = l2.next
        # move the current node to the next node
        curr = curr.next
    # if there are remaining nodes in the first list
    if l1:
        # set the next node to the current node
        curr.next = l1
    # if there are remaining nodes in the second list
    if l2:
        # set the next node to the current node
        curr.next = l2
    # return the merged list
    return dummy.next

def is_palindrome(head):
    """
    Check if a linked list is a palindrome
    """
    # create a list to store the values of the nodes
    values = []
    # iterate through the list
    while head:
        # add the value of the node to the list
        values.append(head.val)
        # move to the next node
        head = head.next
    # compare the list with its reverse
    return values == values[::-1]

def reorder_list(head):
    """
    Reorder a linked list
    """
    # create a list to store the nodes
    nodes = []
    # iterate through the list
    while head:
        # add the node to the list
        nodes.append(head)
        # move to the next node
        head = head.next
    # set the left and right pointers
    left = 0
    right = len(nodes) - 1
    # iterate through the list
    while left < right:
        # set the next pointers for the left and right nodes
        nodes[left].next = nodes[right]
        nodes[right].next = nodes[left + 1]
        # move the left pointer to the right
        left += 1
        # move the right pointer to the left
        right -= 1
    # set the next pointer of the last node to None
    nodes[left].next = None

    # return the head of the reordered list
    return nodes[0]

def get_intersection_node(headA, headB):
    """
    Find the intersection node of two linked lists
    """
    # create sets to store the nodes
    nodes = set()
    # iterate through the first list
    while headA:
        # add the node to the set
        nodes.add(headA)
        # move to the next node
        headA = headA.next
    # iterate through the second list
    while headB:
        # if the node is in the set, return it
        if headB in nodes:
            return headB
        # move to the next node
        headB = headB.next
    # if no intersection is found, return None
    return None

def remove_duplicates(head):
    """
    Remove duplicates from a sorted linked list
    """
    # set the current node to the head of the list
    curr = head
    # iterate through the list
    while curr and curr.next:
        # if the value of the current node is equal to the value of the next node
        if curr.val == curr.next.val:
            # remove the next node
            curr.next = curr.next.next
        # otherwise, move to the next node
        else:
            curr = curr.next
    # return the head of the list
    return head