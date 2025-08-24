from typing import List, Any, Optional, Union
from itertools import chain, combinations


def get_all_combinations(list_of_items: List[Any]) -> List[List[str]]:
    """
    Extract all combinatioins.
    """
    item_num = len(list_of_items)
    all_combinations = list(chain.from_iterable(combinations(list_of_items, r) for r in range(1, item_num + 1)))
    all_combinations.reverse() # reverse the list such that larger combinations comes first
    return all_combinations

def get_combinations(list_of_items: List[Any], samples:Optional[Union[str, List[int]]]= 'all') -> List[List[str]]:
    """"
    Extract combinations

    Arguments:
    list_of_items(List[Any]): list of items whose combinations will be computed
    samples(List(int)): List of integers for sample counts
    """
    if isinstance(samples, str) and samples.lower() == 'all':
        return get_all_combinations(list_of_items)
    elif isinstance(samples, list):
        # check if all sample counts are integer or not 
        if not all(isinstance(n, int) for n in samples):
            raise ValueError("All elements in 'samples' must be integers.")
        
        list_of_combinations=list(chain.from_iterable(combinations(list_of_items, n) for n in samples))
        list_of_combinations.reverse()
        return list_of_combinations
    else:
        raise ValueError("Wrong value for 'sample' count. Enter either all or an integer.")

