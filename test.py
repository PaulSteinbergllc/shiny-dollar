import requests
import json

class NDCApi:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/drugsfda.json"

    def convert_11_to_10_digit_ndcs(self, ndc: str) -> list:
        """Convert 11-digit NDC (5-4-2) to possible 10-digit formats (4-4-2, 5-3-2, 5-4-1)"""
        # Remove any dashes and ensure 11 digits
        ndc_clean = ndc.replace('-', '').replace(' ', '').zfill(11)
        
        formats = []
        
        # 4-4-2 format: Remove leading zero from labeler code
        if ndc_clean[0] == '0':
            formats.append(f"{ndc_clean[1:5]}-{ndc_clean[5:9]}-{ndc_clean[9:11]}")
        
        # 5-3-2 format: Remove zero from product code
        if ndc_clean[5] == '0':
            formats.append(f"{ndc_clean[:5]}-{ndc_clean[6:9]}-{ndc_clean[9:11]}")
        
        # 5-4-1 format: Remove zero from package code
        if ndc_clean[9] == '0':
            formats.append(f"{ndc_clean[:5]}-{ndc_clean[5:9]}-{ndc_clean[10]}")
        
        return formats

    def get_medication_info(self, ndc: str) -> dict:
        """Get medication name and strength from FDA API"""
        try:
            possible_formats = self.convert_11_to_10_digit_ndcs(ndc)
            
            for ndc_format in possible_formats:
                url = f"{self.base_url}?search=openfda.package_ndc:\"{ndc_format}\"&limit=1"
                response = requests.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        result = data['results'][0]
                        if result.get('products'):
                            product = result['products'][0]
                            if product.get('active_ingredients'):
                                active_ingredient = product['active_ingredients'][0]
                                return {
                                    "name": product.get('brand_name', 'Name not found'),
                                    "strength": active_ingredient.get('strength', 'Strength not found')
                                }
            
            return {
                "name": "Name not found",
                "strength": "Strength not found"
            }
                
        except Exception as e:
            logger.error(f"Error fetching medication info: {e}")
            return {
                "name": "Error",
                "strength": "Error"
            }

# Test code
def test_api():
    api = NDCApi()
    test_ndcs = [
        "00006-3026-04",
        "64764-0300-20",
        "50242-0150-01"
    ]
    
    for ndc in test_ndcs:
        print(f"\nTesting NDC: {ndc}")
        info = api.get_medication_info(ndc)
        print(f"Name: {info['name']}")
        print(f"Strength: {info['strength']}")

if __name__ == "__main__":
    test_api()
