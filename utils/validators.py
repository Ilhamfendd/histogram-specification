import os


def validate_image(file_path):
    if not file_path:
        return False, "File path kosong"
    
    if not os.path.exists(file_path):
        return False, "File tidak ditemukan"
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() not in valid_extensions:
        return False, f"Format file tidak didukung. Gunakan: {', '.join(valid_extensions)}"
    
    return True, "Valid"


def validate_output_path(file_path):
    if not file_path:
        return False, "File path kosong"
    
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        return False, "Direktori tidak ditemukan"
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() not in valid_extensions:
        return False, f"Format file tidak didukung. Gunakan: {', '.join(valid_extensions)}"
    
    return True, "Valid"
