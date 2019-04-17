class Solution:
    """
    @param s: the maximum length of s is 1000
    @return: the longest palindromic subsequence's length
    """

    def longestPalindromeSubseq(self, s):
        # write your code here
        length = len(s)
        if length == 0:
            return 0
        longest_palindromic = [[0 for _ in range(length)] for __ in range(length)]
        for j in range(0, length):
            longest_palindromic[j][j] = 1
            for i in range(j - 1, -1, -1):

                if s[i] == s[j]:
                    longest_palindromic[i][j] = 2 + longest_palindromic[i + 1][j - 1]
                else:
                    longest_palindromic[i][j] = max(longest_palindromic[i - 1][j], longest_palindromic[i][j - 1])
        return longest_palindromic[0][length - 1]
s = Solution()

st = 'a'
s.longestPalindromeSubseq(st)