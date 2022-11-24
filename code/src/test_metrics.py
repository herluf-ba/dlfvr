import unittest
from metrics import intersection_over_union


class TestMetrics(unittest.TestCase):

    def test_full_overlap(self):
        b1 = [1.0, 1.0, 2.0, 2.0]
        b2 = [1.0, 1.0, 2.0, 2.0]
        self.assertEqual(intersection_over_union(b1, b2), 1.0)

    def test_no_overlap(self):
        b1 = [1.0, 1.0, 2.0, 2.0]
        b2 = [3.0, 3.0, 5.0, 5.0]
        self.assertEqual(intersection_over_union(b1, b2), 0.0)

    def test_half_overlap(self):
        b1 = [0.0, 0.0, 2.0, 2.0]
        b2 = [0.0, 0.0, 1.0, 1.0]
        self.assertEqual(intersection_over_union(b1, b2), 0.5)


if __name__ == '__main__':
    unittest.main()
