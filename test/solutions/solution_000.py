def flatten(nested):
    """把任意层级的嵌套列表拍平成一维列表。

    例：flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    """
    result = []
    stack = [nested]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            result.append(item)
    return result
