import cv2
import numpy as np


class HistogramProcessor:
    
    @staticmethod
    def histogram_spesifikasi(img_input, img_reference, use_rgb=False):
        if not use_rgb:
            gray_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            gray_reference = cv2.cvtColor(img_reference, cv2.COLOR_BGR2GRAY)
            
            hist_input = cv2.calcHist([gray_input], [0], None, [256], [0, 256])
            hist_reference = cv2.calcHist([gray_reference], [0], None, [256], [0, 256])
            
            hist_input = hist_input / hist_input.sum()
            hist_reference = hist_reference / hist_reference.sum()
            
            cdf_input = np.cumsum(hist_input)
            cdf_reference = np.cumsum(hist_reference)
            
            mapping_table = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                diff = np.abs(cdf_reference - cdf_input[i])
                mapping_table[i] = np.argmin(diff)
            
            gray_result = cv2.LUT(gray_input, mapping_table)
            img_result_bgr = cv2.cvtColor(gray_result, cv2.COLOR_GRAY2BGR)
            
            return img_result_bgr, gray_result, [hist_input], [cv2.calcHist([gray_result], [0], None, [256], [0, 256]) / gray_result.size]
        
        else:
            result = np.zeros_like(img_input, dtype=np.uint8)
            hist_inputs = []
            hist_outputs = []
            
            for i in range(3):
                channel_input = img_input[:, :, i]
                channel_reference = img_reference[:, :, i]
                
                hist_input = cv2.calcHist([channel_input], [0], None, [256], [0, 256])
                hist_reference = cv2.calcHist([channel_reference], [0], None, [256], [0, 256])
                
                hist_input = hist_input / hist_input.sum()
                hist_reference = hist_reference / hist_reference.sum()
                
                hist_inputs.append(hist_input)
                
                cdf_input = np.cumsum(hist_input)
                cdf_reference = np.cumsum(hist_reference)
                
                mapping_table = np.zeros(256, dtype=np.uint8)
                for j in range(256):
                    diff = np.abs(cdf_reference - cdf_input[j])
                    mapping_table[j] = np.argmin(diff)
                
                channel_result = cv2.LUT(channel_input, mapping_table)
                result[:, :, i] = channel_result
                
                hist_out = cv2.calcHist([channel_result], [0], None, [256], [0, 256])
                hist_out = hist_out / hist_out.sum()
                hist_outputs.append(hist_out)
            
            gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            return result, gray_result, hist_inputs, hist_outputs
    
    @staticmethod
    def histogram_equalization(img_input, use_rgb=False):
        if not use_rgb:
            gray_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
            gray_result = cv2.equalizeHist(gray_input)
            
            hist_input = cv2.calcHist([gray_input], [0], None, [256], [0, 256])
            hist_input = hist_input / hist_input.sum()
            
            hist_output = cv2.calcHist([gray_result], [0], None, [256], [0, 256])
            hist_output = hist_output / hist_output.sum()
            
            img_result_bgr = cv2.cvtColor(gray_result, cv2.COLOR_GRAY2BGR)
            
            return img_result_bgr, gray_result, [hist_input], [hist_output]
        
        else:
            result = np.zeros_like(img_input, dtype=np.uint8)
            hist_inputs = []
            hist_outputs = []
            
            for i in range(3):
                channel_input = img_input[:, :, i]
                channel_result = cv2.equalizeHist(channel_input)
                result[:, :, i] = channel_result
                
                hist_input = cv2.calcHist([channel_input], [0], None, [256], [0, 256])
                hist_input = hist_input / hist_input.sum()
                hist_inputs.append(hist_input)
                
                hist_output = cv2.calcHist([channel_result], [0], None, [256], [0, 256])
                hist_output = hist_output / hist_output.sum()
                hist_outputs.append(hist_output)
            
            gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            return result, gray_result, hist_inputs, hist_outputs
