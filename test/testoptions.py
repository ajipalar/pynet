"""
Module for defining testing alternative behavior
"""


def decorate_print_test_results(test, modname):
    """
    Decorate a funciton to add printing
    """

    def internal():
        print(f"Testing {test.__name__} in {modname}")
        test()
        print("Passed!")

    return internal
