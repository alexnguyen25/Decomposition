import unittest
from unittest.mock import patch
from src.preprocessing.validation import validAudio

import numpy

class test_validation(unittest.TestCase):

    @patch('src.preprocessing.validation.librosa.load')
    def test_valid_input(self, mock_load):
        mock_load.return_value = (numpy.zeros(5292000), 44100)
        validAudio('fake.wav')



if __name__ == '__main__':
    unittest.main()
