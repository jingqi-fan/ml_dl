def checkSubarraySum(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: bool
    """
    d = {0: -1}
    d2 = {0: -1}
    s = 0

    for i in range(len(nums)):
        s += nums[i]
        print(f'i {i}, s {s}, d2 {d2}')
        if s in d2 and i - d2[s] > 1:
            return True

        if s not in d2:
            d2[s] = i

        if s % k in d and i - d[s % k] > 1:
            return True
        else:
            d[s % k] = i
    return False
print(checkSubarraySum([5,0,0,0], 3))