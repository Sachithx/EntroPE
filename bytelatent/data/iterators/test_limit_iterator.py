from bytelatent.data.iterators.dev_iterators import BltTestIterator
from bytelatent.data.iterators.limit_iterator import LimitIterator


def test_limit_iterator():
    total = 10
    limit = 5
    base_iterator = BltTestIterator(total=total)
    limit_iterator = LimitIterator(base_iterator, limit=limit)
    iterator = limit_iterator.create_iter()
    n = 0
    for example in iterator:
        assert example.sample_id == f"test_{n}"
        n += 1
    assert n == limit

    limit = 10
    base_iterator = BltTestIterator(total=total)
    limit_iterator = LimitIterator(base_iterator, limit=limit)
    iterator = limit_iterator.create_iter()
    n = 0
    for example in iterator:
        assert example.sample_id == f"test_{n}"
        n += 1
    assert n == limit == total

    limit = 20
    base_iterator = BltTestIterator(total=total)
    limit_iterator = LimitIterator(base_iterator, limit=limit)
    iterator = limit_iterator.create_iter()
    n = 0
    for example in iterator:
        assert example.sample_id == f"test_{n}"
        n += 1
    assert n == total

    limit = -1
    base_iterator = BltTestIterator(total=total)
    limit_iterator = LimitIterator(base_iterator, limit=limit)
    iterator = limit_iterator.create_iter()
    n = 0
    for example in iterator:
        assert example.sample_id == f"test_{n}"
        n += 1
    assert n == total
