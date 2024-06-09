#create function on commonly asked question on strings
def palindrome(s):
    """
    Check if a string is a palindrome
    """
    # remove non-alphanumeric characters and convert to lowercase
    s = ''.join(char for char in s if char.isalnum()).lower()
    # compare the string with its reverse
    return s == s[::-1]

def reverse_words(s):
    """
    Reverse the words in a string
    """
    # split the string into words
    words = s.split()
    # reverse the words and join them back together
    return ' '.join(words[::-1])

def is_anagram(s, t):
    """
    Check if two strings are anagrams
    """
    # sort the characters in each string and compare
    return sorted(s) == sorted(t)

def longest_substring(s):
    """
    Find the length of the longest substring without repeating characters
    """
    # create a dictionary to store the index of each character
    char_index = {}
    # initialize the start of the substring and the maximum length
    start = 0
    max_length = 0
    # iterate through the string
    for i, char in enumerate(s):
        # if the character is in the dictionary and its index is after the start of the substring
        if char in char_index and char_index[char] >= start:
            # update the start of the substring
            start = char_index[char] + 1
        # update the index of the character
        char_index[char] = i
        # update the maximum length
        max_length = max(max_length, i - start + 1)
    # return the maximum length
    return max_length

def longest_palindromic_substring(s):
    """
    Find the longest palindromic substring in a string
    """
    # initialize variables to store the start and end of the longest palindrome
    start = 0
    end = 0
    # iterate through the string
    for i in range(len(s)):
        # find the length of the palindrome centered at the current character
        len1 = expand_around_center(s, i, i)
        # find the length of the palindrome centered between the current and next characters
        len2 = expand_around_center(s, i, i + 1)
        # find the maximum length palindrome
        max_len = max(len1, len2)
        # if the length of the palindrome is greater than the current maximum
        if max_len > end - start:
            # update the start and end of the longest palindrome
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    # return the longest palindrome
    return s[start:end + 1]

def expand_around_center(s, left, right):
    """
    Helper function to find the length of a palindrome centered at two characters
    """
    # while the characters match and the indices are within the bounds of the string
    while left >= 0 and right < len(s) and s[left] == s[right]:
        # expand the palindrome
        left -= 1
        right += 1
    # return the length of the palindrome
    return right - left - 1

def longest_common_prefix(strs):
    """
    Find the longest common prefix among a list of strings
    """
    # if the list is empty, return an empty string
    if not strs:
        return ''
    # sort the list of strings
    strs.sort()
    # compare the first and last strings in the list
    first = strs[0]
    last = strs[-1]
    # find the common prefix between the first and last strings
    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1
    return first[:i]

def group_anagrams(strs):
    """
    Group anagrams together in a list of strings
    """
    # create a dictionary to store the anagrams
    anagrams = {}
    # iterate through the strings
    for s in strs:
        # sort the characters in the string
        sorted_s = ''.join(sorted(s))
        # add the string to the list of anagrams
        if sorted_s in anagrams:
            anagrams[sorted_s].append(s)
        else:
            anagrams[sorted_s] = [s]
    # return the list of anagrams
    return list(anagrams.values())

def valid_palindrome(s):
    """
    Check if a string is a valid palindrome
    """
    # remove non-alphanumeric characters and convert to lowercase
    s = ''.join(char for char in s if char.isalnum()).lower()
    # compare the string with its reverse
    return s == s[::-1]

