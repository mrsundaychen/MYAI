"""代码生成评测集：函数级任务，agent 补全函数实现，用单测判分。

每个任务 = prompt（函数签名 + docstring 规格）+ tests（单测）。
判分：把 agent 生成的代码和 tests 拼起来跑，全过 = pass。
"""

TASKS = [
    {
        "task_id": "gen-001",
        "prompt": '''def flatten(nested):
    """把任意层级的嵌套列表拍平成一维列表。

    例：flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    """
''',
        "entry_point": "flatten",
        "tests": '''def test():
    assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    assert flatten([]) == []
    assert flatten([[[]]]) == []
''',
    },
    {
        "task_id": "gen-002",
        "prompt": '''def fib(n):
    """返回第 n 个斐波那契数（F0=0, F1=1）。n >= 0。"""
''',
        "entry_point": "fib",
        "tests": '''def test():
    assert fib(0) == 0
    assert fib(1) == 1
    assert fib(10) == 55
    assert fib(20) == 6765
''',
    },
    {
        "task_id": "gen-003",
        "prompt": '''def is_palindrome(s):
    """判断字符串是否回文（忽略大小写）。

    例：is_palindrome("racecar") == True
    """
''',
        "entry_point": "is_palindrome",
        "tests": '''def test():
    assert is_palindrome("racecar") is True
    assert is_palindrome("hello") is False
    assert is_palindrome("Aba") is True
    assert is_palindrome("") is True
''',
    },
    {
        "task_id": "gen-004",
        "prompt": '''def dedup(lst):
    """去掉列表里的重复元素，保持原有顺序。

    例：dedup([1, 2, 2, 3, 1]) == [1, 2, 3]
    """
''',
        "entry_point": "dedup",
        "tests": '''def test():
    assert dedup([1, 2, 2, 3, 1]) == [1, 2, 3]
    assert dedup([]) == []
    assert dedup(["a", "a", "b"]) == ["a", "b"]
''',
    },
    {
        "task_id": "gen-005",
        "prompt": '''def chunk(lst, n):
    """把列表按每块 n 个元素切分，返回列表的列表。

    例：chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    """
''',
        "entry_point": "chunk",
        "tests": '''def test():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk([], 3) == []
    assert chunk([1], 1) == [[1]]
''',
    },
    {
        "task_id": "gen-006",
        "prompt": '''def sum_digits(n):
    """返回非负整数 n 各位数字之和。

    例：sum_digits(123) == 6
    """
''',
        "entry_point": "sum_digits",
        "tests": '''def test():
    assert sum_digits(123) == 6
    assert sum_digits(0) == 0
    assert sum_digits(1009) == 10
''',
    },
    {
        "task_id": "gen-007",
        "prompt": '''def reverse_words(s):
    """反转字符串里的单词顺序（单词之间以单个空格分隔）。

    例：reverse_words("hello world") == "world hello"
    """
''',
        "entry_point": "reverse_words",
        "tests": '''def test():
    assert reverse_words("hello world") == "world hello"
    assert reverse_words("a b c") == "c b a"
    assert reverse_words("single") == "single"
''',
    },
    {
        "task_id": "gen-008",
        "prompt": '''def parse_kv(s):
    """把 "k1=v1,k2=v2" 形式的字符串解析成 dict。

    例：parse_kv("a=1,b=2") == {"a": "1", "b": "2"}
    """
''',
        "entry_point": "parse_kv",
        "tests": '''def test():
    assert parse_kv("a=1,b=2") == {"a": "1", "b": "2"}
    assert parse_kv("") == {}
    assert parse_kv("x=y") == {"x": "y"}
''',
    },
    {
        "task_id": "gen-009",
        "prompt": '''def most_common(lst):
    """返回列表里出现次数最多的元素；并列时返回先出现的那一个。

    例：most_common([1, 2, 2, 3]) == 2
    """
''',
        "entry_point": "most_common",
        "tests": '''def test():
    assert most_common([1, 2, 2, 3]) == 2
    assert most_common(["a", "b", "a"]) == "a"
    assert most_common([5]) == 5
''',
    },
    {
        "task_id": "gen-010",
        "prompt": '''def group_by(lst, key):
    """按 key 函数对列表元素分组，返回 {键: 元素列表}。

    例：group_by([1, 2, 3, 4], lambda x: x % 2) == {1: [1, 3], 0: [2, 4]}
    """
''',
        "entry_point": "group_by",
        "tests": '''def test():
    assert group_by([1, 2, 3, 4], lambda x: x % 2) == {1: [1, 3], 0: [2, 4]}
    assert group_by([], lambda x: x) == {}
    assert group_by(["aa", "b", "cc"], len) == {2: ["aa", "cc"], 1: ["b"]}
''',
    },
]
