import cv2
from PIL import Image, ImageTk


class ImageHandler:
    
    @staticmethod
    def load_image(file_path):
        try:
            img = cv2.imread(file_path)
            if img is None:
                return None, False
            return img, True
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, False
    
    @staticmethod
    def resize_image(img, max_size=1000):
        height, width = img.shape[:2]
        
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return img
    
    @staticmethod
    def display_on_canvas(img, canvas, canvas_size=300):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width = img_rgb.shape[:2]
        
        if width > height:
            new_width = canvas_size
            new_height = int(height * (canvas_size / width))
        else:
            new_height = canvas_size
            new_width = int(width * (canvas_size / height))
        
        img_resized = cv2.resize(img_rgb, (new_width, new_height))
        
        pil_image = Image.fromarray(img_resized)
        photo = ImageTk.PhotoImage(pil_image)
        
        canvas.create_image(canvas_size // 2, canvas_size // 2, image=photo)
        canvas.image = photo
    
    @staticmethod
    def save_image(file_path, img):
        try:
            cv2.imwrite(file_path, img)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
