import os 
from PIL import Image

def check_and_remove_corrupted_image(file_path : str):
    try:
        with Image.open(file_path) as img:
            img.verify()  
        return False 
    except (IOError, SyntaxError) as e:
        print(f"Removing corrupted image: {file_path} - {e}")
        os.remove(file_path)  
        return True 
    
def scan_the_dir(path : str):
    for root,directory,file in os.walk(path):
        for label in directory:
            print(f"Checking the {label} Directory")
            for file in os.listdir(os.path.join(root,label)):
                check_and_remove_corrupted_image(os.path.join(root,label,file))
            print(f"Completed checking the {label} Directory")
            
print("Checking the Train Directory")
scan_the_dir(r"Dataset\Extracted\train")
print("-------------------------------")
print("Checking the Val Directory")
scan_the_dir(r"Dataset\Extracted\val")