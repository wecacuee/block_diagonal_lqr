import re


def common_substr(strs, return_diffs=False):
    """
    >>> common_substr("abcd", "xbcd")
    'bcd'
    """
    strs = list(strs)
    min_len = min(map(len, strs))
    max_len = max(map(len, strs))
    first = strs[0]
    comm = type(first)()
    diffs = [type(first)() for s in strs]
    for i in range(max_len):
        if i < min_len and all((first[i] == s[i]) for s in strs):
            comm.append(first[i])
        else:
            for d, s in zip(diffs, strs):
                if i < len(s):
                    d.append(s[i])
    return (comm, diffs) if return_diffs else comm


def diff_substr(strs, splitre="[-_/]", joinstr="-"):
    comm, diffs = common_substr((re.split(splitre, s) for s in strs),
                                return_diffs=True)
    return map(joinstr.join, diffs)
