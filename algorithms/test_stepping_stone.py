import unittest

from stepping_stone import stepping_stone_method

class SteppingStoneMethodTest(unittest.TestCase):
    def test_stepping_stone_method(self):
        # Test case 1
        costs = [[2, 3, 4],
                 [5, 6, 7],
                 [8, 9, 10]]
        supply = [10, 20, 30]
        demand = [15, 25, 35]
        expected_allocations = [[10, 0, 0],
                                [5, 15, 0],
                                [0, 10, 20]]
        self.assertEqual(stepping_stone_method(costs, supply, demand), expected_allocations)

        # Test case 2
        costs = [[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]]
        supply = [10, 20, 30]
        demand = [15, 25, 35]
        expected_allocations = [[10, 0, 0],
                                [5, 15, 0],
                                [0, 10, 20]]
        self.assertEqual(stepping_stone_method(costs, supply, demand), expected_allocations)

        # Test case 3
        costs = [[1, 2],
                 [3, 4]]
        supply = [10, 20]
        demand = [15, 25]
        expected_allocations = [[10, 0],
                                [5, 15]]
        self.assertEqual(stepping_stone_method(costs, supply, demand), expected_allocations)

if __name__ == '__main__':
    unittest.main()