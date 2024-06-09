
def longest_common_subsequence(s1, s2):
    """
    Find the length of the longest common subsequence between two strings
    """
    # create a 2D array to store the lengths of common subsequences
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    # iterate through the strings
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            # if the characters match, update the length of the common subsequence
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            # otherwise, the length of the common subsequence is the maximum of the previous characters
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # return the length of the longest common subsequence
    return dp[-1][-1]

def largest_common_substring(s1, s2):
    """
    Find the length of the largest common substring between two strings
    """
    # create a 2D array to store the lengths of common substrings
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    # initialize the maximum length of a common substring
    max_len = 0
    # iterate through the strings
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            # if the characters match, update the length of the common substring
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
    # return the maximum length of a common substring
    return max_len

def longest_increasing_subsequence(nums):
    """
    Find the length of the longest increasing subsequence in an array of integers
    """
    # create a list to store the lengths of increasing subsequences
    dp = [1] * len(nums)
    # iterate through the array
    for i in range(1, len(nums)):
        # iterate through the previous elements
        for j in range(i):
            # if the current element is greater than the previous element, update the length of the subsequence
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    # return the maximum length of an increasing subsequence
    return max(dp)

def edit_distance(s1, s2):
    """
    Find the minimum number of operations required to convert one string into another
    """
    # create a 2D array to store the edit distances
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    # initialize the edit distances for empty strings
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    # iterate through the strings
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            # if the characters match, the edit distance is the same as the previous characters
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            # otherwise, the edit distance is the minimum of the previous characters plus one
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    # return the edit distance between the two strings
    return dp[-1][-1]

def coin_change(coins, amount):
    """
    Find the minimum number of coins needed to make up a given amount
    """
    # create a list to store the minimum number of coins for each amount
    dp = [float('inf')] * (amount + 1)
    # the minimum number of coins for amount 0 is 0
    dp[0] = 0
    # iterate through the amounts
    for i in range(1, amount + 1):
        # iterate through the coins
        for coin in coins:
            # if the coin value is less than or equal to the amount
            if coin <= i:
                # update the minimum number of coins for the amount
                dp[i] = min(dp[i], dp[i - coin] + 1)
    # return the minimum number of coins for the given amount
    return dp[amount]

def max_product_subarray(nums):
    """
    Find the maximum product of a subarray in an array of integers
    """
    # initialize variables to store the maximum and minimum products
    max_product = nums[0]
    min_product = nums[0]
    result = nums[0]
    # iterate through the array
    for i in range(1, len(nums)):
        # calculate the maximum and minimum products
        if nums[i] < 0:
            max_product, min_product = min_product, max_product
        max_product = max(nums[i], max_product * nums[i])
        min_product = min(nums[i], min_product * nums[i])
        # update the result
        result = max(result, max_product)
    # return the maximum product
    return result

def knapsack(weights, values, capacity):
    """
    Find the maximum value that can be obtained by filling a knapsack with a given capacity
    """
    # create a 2D array to store the maximum value for each weight and capacity
    dp = [[0] * (capacity + 1) for _ in range(len(weights) + 1)]
    # iterate through the weights
    for i in range(1, len(weights) + 1):
        # iterate through the capacities
        for j in range(1, capacity + 1):
            # if the weight is less than or equal to the capacity
            if weights[i - 1] <= j:
                # update the maximum value
                dp[i][j] = max(dp[i - 1][j], values[i - 1] + dp[i - 1][j - weights[i - 1]])
            # otherwise, the maximum value is the same as the previous weight
            else:
                dp[i][j] = dp[i - 1][j]
    # return the maximum value that can be obtained
    return dp[-1][-1]

def unbounded_knapsack(weights, values, capacity):
    """
    Find the maximum value that can be obtained by filling a knapsack with a given capacity
    """
    # create a list to store the maximum value for each capacity
    dp = [0] * (capacity + 1)
    # iterate through the capacities
    for i in range(1, capacity + 1):
        # iterate through the weights
        for j in range(len(weights)):
            # if the weight is less than or equal to the capacity
            if weights[j] <= i:
                # update the maximum value
                dp[i] = max(dp[i], values[j] + dp[i - weights[j]])
    # return the maximum value that can be obtained
    return dp[-1]

def minimum_path_sum(grid):
    """
    Find the minimum path sum in a grid from the top-left to the bottom-right
    """
    # create a 2D array to store the minimum path sum for each cell
    dp = [[0] * len(grid[0]) for _ in range(len(grid))]
    # set the initial value for the top-left cell
    dp[0][0] = grid[0][0]
    # set the initial values for the first row and column
    for i in range(1, len(grid)):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, len(grid[0])):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    # iterate through the grid
    for i in range(1, len(grid)):
        for j in range(1, len(grid[0])):
            # update the minimum path sum for each cell
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    # return the minimum path sum for the bottom-right cell
    return dp[-1][-1]

def maximum_subarray(nums):
    """
    Find the maximum subarray sum in an array of integers
    """
    # initialize variables to store the current sum and maximum sum
    current_sum = 0
    max_sum = float('-inf')
    # iterate through the array
    for num in nums:
        # calculate the current sum
        current_sum = max(num, current_sum + num)
        # update the maximum sum
        max_sum = max(max_sum, current_sum)
    # return the maximum sum
    return max_sum