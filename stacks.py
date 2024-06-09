
def is_valid_parentheses(s):
    """
    Check if a string of parentheses is valid
    """
    # create a stack to store the opening parentheses
    stack = []
    # create a dictionary to store the matching parentheses
    mapping = {')': '(', '}': '{', ']': '['}
    # iterate through the string
    for char in s:
        # if the character is an opening parenthesis, push it onto the stack
        if char in mapping.values():
            stack.append(char)
        # if the character is a closing parenthesis
        elif char in mapping.keys():
            # if the stack is empty or the top of the stack does not match the closing parenthesis, return False
            if not stack or stack.pop() != mapping[char]:
                return False
    # if the stack is empty, return True
    return not stack

def evaluate_postfix(expression):
    """
    Evaluate a postfix expression
    """
    # create a stack to store the operands
    stack = []
    # iterate through the expression
    for char in expression:
        # if the character is a digit, push it onto the stack
        if char.isdigit():
            stack.append(int(char))
        # if the character is an operator
        else:
            # pop the top two operands from the stack
            operand2 = stack.pop()
            operand1 = stack.pop()
            # perform the operation and push the result onto the stack
            if char == '+':
                stack.append(operand1 + operand2)
            elif char == '-':
                stack.append(operand1 - operand2)
            elif char == '*':
                stack.append(operand1 * operand2)
            elif char == '/':
                stack.append(operand1 // operand2)
    # return the final result
    return stack.pop()

def next_greater_element(nums1, nums2):
    """
    Find the next greater element for each element in nums1 in nums2
    """
    # create a dictionary to store the next greater element for each element in nums2
    next_greater = {}
    # create a stack to store the elements
    stack = []
    # iterate through the elements in nums2
    for num in nums2:
        # while the stack is not empty and the current element is greater than the top of the stack
        while stack and num > stack[-1]:
            # pop the top of the stack and store it as the next greater element for the current element
            next_greater[stack.pop()] = num
        # push the current element onto the stack
        stack.append(num)
    # iterate through the elements in nums1
    for i in range(len(nums1)):
        # set the next greater element for each element in nums1
        nums1[i] = next_greater.get(nums1[i], -1)
    # return the result
    return nums1

def largest_rectangle_area(heights):
    """
    Find the largest rectangle area in a histogram
    """
    # create a stack to store the indices of the heights
    stack = []
    # set the maximum area to 0
    max_area = 0
    # iterate through the heights
    for i in range(len(heights)):
        # while the stack is not empty and the current height is less than the height at the top of the stack
        while stack and heights[i] < heights[stack[-1]]:
            # calculate the height of the rectangle
            height = heights[stack.pop()]
            # calculate the width of the rectangle
            width = i if not stack else i - stack[-1] - 1
            # calculate the area of the rectangle
            max_area = max(max_area, height * width)
        # push the current index onto the stack
        stack.append(i)
    # iterate through the remaining heights in the stack
    while stack:
        # calculate the height of the rectangle
        height = heights[stack.pop()]
        # calculate the width of the rectangle
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        # calculate the area of the rectangle
        max_area = max(max_area, height * width)
    # return the maximum area
    return max_area