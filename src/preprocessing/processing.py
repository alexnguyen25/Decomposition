import os


def checkFileFormat(file_path):
    """
    Checks the file format and for corruption

    Args:
        file_path: The file path of the audio file.
    
    Returns:
        bool: Whether or not the file is corrupted or is a wav.
    """
    
    filename, ext = os.path.splitext(file_path)

    return ext == 'wav'



