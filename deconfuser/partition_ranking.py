import numpy as np

def get_ranked_partitions(possible_groups):
    """
    The partitions of groups are ranked such that the top partition has the largest first group (a trajectory corresponding to most observations), the largest second group from the remaining observations, etc.
    partitions that are "included" in partitions with more grouped observations, are excluded.

    Parameters
    ----------
    possible_groups : np.array of floats of shape (?,2)
        positions of planet candidates in image plane

    Returns
    -------
    an iterator over
        list of lists of ints
            lists of indices of observations grouped together by belonging to the same planet
    """

    filtered_partitions = []
    for partition in get_ranked_set_partitions(possible_groups):
        if not any(_is_sub_partition(partition,other_partition) for other_partition in filtered_partitions):
            #there if no previous partition that already includes this one (with even more planets matched)
            filtered_partitions.append(partition)
            yield list(map(sorted, partition))

def _get_number_partitions_up_to_sum(ranked_numbers, sum_):
    """
    Iterator over all possible partitions (with repetitions) of numbers that sum up to a target sum (larger numbers appear first).
    Used by get_ranked_set_partitions.
    (note that in the contect of planet matching, 1 is always in ranked_numbers - representing a single observation of a planet)

    Parameters
    ----------
    ranked_numbers : list of ints
        list of numbers to be summed (ranked in increasing order)
    sum_ : int
        target sum

    Returns
    -------
    an iterator over
        tuple of ints
            partitions from ranked_numbers in decreasing lexical order
    """
    #index of the highest number that is lower than the target sum
    i = np.searchsorted(ranked_numbers, sum_, side="right")
    #if all number are higher, yield an empty partition and return to end the recursion
    if i == 0:
        yield ()
        return

    #iterate over numbers smaller than the target sum in decreasing order
    for j in range(i-1,-1,-1):
        #iterate over partitions of numbers summping up to the remainder (after the largest number is accounted for)
        for p in _get_number_partitions_up_to_sum(ranked_numbers[:j+1], sum_ - ranked_numbers[j]):
            #yield the partitions including the largest number
            yield (ranked_numbers[j],) + p

def _get_set_partitions_specific(sets, specific_lengths, excluded_set):
    """
    Iterator over all possible partitions of disjoint subsets with specified lengths
    Used by get_ranked_set_partitions.

    Parameters
    ----------
    sets : list of python sets ranked by length (in decreasing order)
        sets to choose from
    specific_lengths : list of ints
        the lenghts that returned subsets should be of

    Returns
    -------
    an iterator over
        tuple sets
            disjoint subsets (same number of elements as specific_lengths)
    """
    #if all data points were processed yield an empty partition and terminate recursion
    if len(specific_lengths) == 0:
        yield ()
        return

    #iterate over all sets with the largest lengths that were not already processed (not in excluded_set)
    for i,set_ in enumerate(sets):
        if len(set_) != specific_lengths[0] or excluded_set.intersection(set_): continue

        #iterate over partitions with the rest of the sets
        for kp in _get_set_partitions_specific(sets[i+1:], specific_lengths[1:], excluded_set|set_):
            #yield the partition with the currently processed set
            yield (set_,) + kp

def get_ranked_set_partitions(sets):
    """
    Iterator over partitions of disjoint subsets from a given list of sets, ranked by subset lengths.
    partitions with larger subsets are yielded first.

    Parameters
    ----------
    sets : list of python sets
        sets to choose from

    Returns
    -------
    an iterator over
        tuple sets
            disjoint subsets
    """
    sets = sorted(map(set, sets), key=len, reverse=True)

    #get all allowable numbers of data points per trajectory (lenghts)
    lengths = sorted(set(map(len, sets)))

    #get partitions of lengths that sum up to the total number of data points
    for specific_lengths in _get_number_partitions_up_to_sum(lengths, len(set.union(*sets))):
        #recursively iterate over set partitions with specific lenghts
        for kp in _get_set_partitions_specific(sets, specific_lengths, set()):
            yield kp

def _is_sub_partition(sets, other_sets):
    """
    Checks if all the sets in one list are subsets of one set in another list.
    Used by get_ranked_trajectory_partitions

    Parameters
    ----------
    sets : list of python sets
        partition to check
    other_sets : list of python sets
        partition to check against

    Returns
    -------
    bool
        whether all sets are included in other_sets
    """
    for set_ in sets:
        if not any(map(set_.issubset, other_sets)):
            #here if at least one set is not fully included in another set in the other partition
            return False
    return True

