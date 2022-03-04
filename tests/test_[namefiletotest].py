
import pytest
#import functions to test from your package


#Exemple of Datasets
DATASETS = {
    "diabetes": {'X': diabetes[0], 'Y': diabetes[1]},
    "iris": {'X': iris[0], 'Y': iris[1]},
    "digits": {'X': digits[0], 'Y': digits[1]},
}

#Exemple of a fixture (to put on the line before the def of a test function)
#to loop a test with parameters (here DATASETS parameters
#@pytest.mark.parametrize("nameSet, dataXY", DATASETS.items())



#For ML : Validation of the datasets
#def test_X_y():
#    """Input validation for standard estimators.
#
#    Checks X and y for consistent length, enforces X to be 2D and y 1D.
#    By default, X is checked to be non-empty and containing only finite values.
#    Standard input checks are also applied to y,
#    such as checking that y does not have np.nan or np.inf targets.
#
#    Returns
#    -------
#    None
#    """
#    check_X_y(X, y)


class TestFunction1():
    """
    Test class for function1
    """

    def test_goodcase1(self):
        """
        """

    def test_goodcase2(self):
        """
        """

    def test_goodcase3(self):
        """
        """

    def test_badcase1(self):
        """
        """

    def test_badcase2(self):
        """
        """

    def test_badcase3(self):
        """
        """
		
	
    def test_specialcase1(self):
        """
        """
	
	def test_specialcase1(self):
        """
        """
		
		
class TestFunction2():
    """
    Test class for function1
    """

    def test_goodcase1(self):
        """
        """

    def test_goodcase2(self):
        """
        """

    def test_goodcase3(self):
        """
        """

    def test_badcase1(self):
        """
        """

    def test_badcase2(self):
        """
        """

    def test_badcase3(self):
        """
        """
		
	
    def test_specialcase1(self):
        """
        """
	
	def test_specialcase1(self):
        """
        """