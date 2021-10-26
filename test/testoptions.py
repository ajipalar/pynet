"""
Module for defining testing alternative behavior
"""

def decorate_print_test_results(test):
    """
    Decorate a funciton to add printing
    """
    def internal():
        print(f'Testing {test.__name__} in {locals()["__name__"]}')
        test()
        print("Passed!")
    return internal
