from typing import List

def findMaxSubArray(A: List[int]) -> List[int]:
    max_sum = -float('infinity')
    max_start_id = 0
    max_end_id = 0
    
    cur_sum = -float('infinity')

    for i, ele in enumerate(A):
        if cur_sum <= 0:
            cur_sum, cur_start_id = 0, i
        
        cur_sum += ele
        cur_end_id = i

        if cur_sum > max_sum:
            max_sum  = cur_sum
            max_start_id, max_end_id = cur_start_id, cur_end_id

    return A[max_start_id: max_end_id+1]