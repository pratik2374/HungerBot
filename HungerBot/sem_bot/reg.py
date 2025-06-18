from to_check_type import classify
import re
from difflib import SequenceMatcher

sample_products = [
    "Cutting chai/coffee",
    "Watermelon Juice(200 ml)",
    "Snacks Combo 1",
    "Bread & Butter Jam",
    "Masala Veg Maggi",
    "Nippatu",
    "Chakli",
    "Mix Fruit Cut Platter(200 Gms)",
    "Bun Maska",
    "Samosa",
    "Coffee",
    "beverages",
    "Snacks",
    "Tea",
    "Breakfast Buffet",
    "Sweet and salt cookies",
    "Breakfast  Buffet (Non-Veg)",
    "Breakfast  Buffet (Veg)",
    "Filter Coffee",
    "Ginger Tea",
    "Beetroot Juice(200 ml)",
    "Cutting Chai/Coffee - 100 ml",
    "Choice of Omelette",
    "Choice of Omelette (Organic Eggs)",
    "Mango Shake (200ml)",
    "Papaya Cut(200 Gms)",
    "Lemon Tea",
    "Chicken Puff",
    "Mix Fruit Juice",
    "Paneer Sandwich with Wafer",
    "Vanilla Muffin",
    "Watermelon Juice",
    "Tender Coconut Water Glass (200ml)",
    "Paneer Sandwich",
    "Veg Sandwich with Wafer",
    "Oats & Raisins Cookies",
    "Lunch Buffet (Veg)",
    "Veg Lunch Buffet",
    "Lunch  Buffet (Non-Veg)",
    "Lunch Buffet - Non Veg",
    "Chocolate Brownie",
    "Tender Coconut Water (With Shell)",
    "Pomegranate Juice(200 ml)",
    "Mosambi Juice (200 ml)",
    "Chicken Biryani",
    "Make your own salad - Veg",
    "Plain Croissant",
    "Buffet NV lunch",
    "Chocolate Croissant",
    "Sweet Bun Maska",
    "Banana Cut (200 Gms)",
    "Tender Coconut water with Shell",
    "Ash Gourd Juice",
    "Choice of Eggs (2 Egg)",
    "Watermelon Cut(200 Gms)",
    "Guava Cut (200 Gms)",
    "Melon/ Sarda Cut(200 Gms)",
    "Banana Shake(200 ml)",
    "Boiled Egg ( 2 Pcs)",
    "Snacks Combo 2",
    "Maggi combo",
    "Blueberry Muffin",
    "Non Veg Sandwich",
    "Apple Cut (200 Gms)",
    "Veg Maggie",
    "Veg Puff (Patty)",
    "Double Choco Chip Cookies",
    "Peanut Butter & Caramel Cookies",
    "Apple Milkshake",
    "Choice of egg with Choice of Bread",
    "Mix Fruit Juice(200 ml)",
    "Dinner  Buffet (Veg)",
    "Dinner Buffet - Veg",
    "Dinner Buffet - Non Veg",
    "Dinner  Buffet (Non-Veg)",
    "Papaya Shake (200 ml)",
    "Chocolate Pastry",
    "Double Chocolate Muffin",
    "Egg Puff",
    "Chicken Sandwich",
    "Papaya Juice",
    "Choice of Omelette ( Organic Egg 2 Nos)",
    "ABC Juice 200ML",
    "Pineapple Pastry",
    "Breakfast Buffet (Non Veg)",
    "Veg Indian BF",
    "Breakfast Indian -Veg",
    "Pineapple Cut",
    "Irani Chai",
    "Breakfast",
    "Masala Tea",
    "Cutting Chai or Coffee 100 ml",
    "Boiled Egg Double",
    "Mix Fruit Platter",
    "Fresh Sugar Cane Juice",
    "North Indian Breakfast",
    "Ginger Tea - 120 ml",
    "Indian Veg Breakfast",
    "Tea / Coffee",
    "Filter Coffee - 120 ml",
    "Mosambi Juice (200 ML)",
    "Breakfast Buffet (Veg)",
    "Vegetable Juice (200 ML)",
    "Mix Fruit Cut 250 gm",
    "Mix Juice (200 ML)",
    "Tender Coconut Water with Shell",
    "Monday Morning Tea",
    "Mix Fruit Cut (250 gm)",
    "Pineapple Juice(200 ml)",
    "Banana Shake (200 ML)",
    "Cutting Tea",
    "Tender Coconut Water (with shell)",
    "Breakfast Indian -Non Veg",
    "Cold Coffee (200 ML)",
    "Mix fruit juice 200ML",
    "Filter Coffee- 120ml",
    "Mosambi- Juice",
    "Mausami Juice (200 ML)",
    "Irani Chai - 120 ml",
    "Buttermilk Masala",
    "Paneer Sandwich Veg",
    "Cheese Maggi",
    "Iced Cafe Mocha 300 Ml",
    "Tender Coconut",
    "Mango Lassi",
    "Cutting Chai/Coffee- 100 ml",
    "Veg Puff",
    "Lemon Tea - 120 ml",
    "Veg Cheese Grilled Sandwich",
    "Mango Shake",
    "NonVeg Indian BF",
    "Morning Breakfast",
    "Badam Milk - 120 ml",
    "Coconut Flavour cookies",
    "Masala Omlets with Bread Slices ( 2 Pcs)",
    "Choice of Dosa (Plan, Masala & Onion)",
    "Masala Butter Milk",
    "Butter Croissant",
    "Chocolate Doughnut",
    "Carrot Juice(200 ml)",
    "Lemon Water Jal Jeera",
    "Veg Sandwich",
    "Coconut Tender with Shell",
    "Sugar Cane Juice (200ml)",
    "Egg Bhurji with Bread Slices ( 2 Pcs)",
    "Sweetlime/Mosambi Juice",
    "Cane Fresh 250Ml",
    "Papaya Shake",
    "Coconut Water",
    "Milk Shake  300 Ml",
    "Capsicum Cheese",
    "Water Melon and Ginger Juice - (The good way)",
    "Veg Indian",
    "Chicken (Chili Chipotle) Bowl",
    "Non Veg Combo",
    "Egg Combo",
    "Non Veg Indian",
    "Veg Combo",
    "Choix Veg Meal",
    "Grill Station Non Veg",
    "Chicken (Crispy Peri Peri) Nachos",
    "Tandoor Combo Non-Veg",
    "Carrot Honey & Whole Wheat Cake",
    "Non Veg Grain Bowl Lunch",
    "Chicken Combo",
    "Classic Margherita Pizza",
    "Wholesome Bowl Veg",
    "Buffet Veg Lunch",
    "Chicken Tikka Pizza",
    "Chole Kulche",
    "Chicken (Chili Chipotle) Burrito",
    "Lunch Buffet (Non-Veg)",
    "Tandoor Combo Veg",
    "Lunch Buffet - Veg",
    "Pineapple Cut(200 Gms)",
    "Chicken (Crispy Peri Peri) Bowl",
    "Pav Bhaji",
    "Vada pav",
    "Asian Veg Bowl",
    "Mushroom (Crispy) Bowl",
    "Veg Biryani",
    "Veg Lunch",
    "Paneer Roll",
    "Tuna Signature Wrap.",
    "Make your own Egg Salad-Egg",
    "Paneer",
    "Asian-Non Veg Bowl",
    "Mini Chicken (Chili Chipotle) Bowl",
    "Choix Non-Veg",
    "Mini Potato (Peri Peri) Bowl",
    "Extra Paneer (BBQ)",
    "Delhi Wale Rajma With Rice",
    "Shikanji",
    "Paneer Sub",
    "Tandoor Starter Veg",
    "Veg Burger",
    "Wholesome Bowl Non-Veg",
    "Choco Lava",
    "Veg Tossed Salad",
    "Tandoor Starter Non-Veg",
    "Tandoor Roti/Naan",
    "Badam Milk Shake  300 Ml",
    "Tandoori Paneer Tikka Pizza",
    "Make your own salad - Non-Veg",
    "Make Your Own Salad Veg",
    "Lime Juice",
    "Rice Veg Curry Combo",
    "Veg Choix Bowl",
    "Non Veg Buffet",
    "Veg Grain Bowl Lunch",
    "Assorted Muffins",
    "Mini Mushroom (Crispy) Salad",
    "Veg Buffet",
    "Veg Biryani Combo",
    "Mint Milk Chocolate Chips Ice cream",
    "Veggie Momo-Supreme",
    "Water Melon and Ginger Juice (The good way)",
    "Mini Potato (Peri Peri) Burrito",
    "Peanut Butter and Caramel Cookies",
    "Chicken Steamed Momos",
    "Grill Station Veg",
    "Kesar Pista Ice cream",
    "Papaya Juice(200 ml)",
    "Paneer (BBQ) Bowl",
    "Non Veg Lunch",
    "Chicken (BBQ) Pro-Salad",
    "Walnut Brownie",
    "Non Veg",
    "Paneer (Mexican) Pro-Bowl",
    "Chicken Starter",
    "Extra Guac",
    "Crushed Corn Chips",
    "Extra Cheese",
    "Extra Corn",
    "Mini Chicken (Crispy Peri Peri) Bowl",
    "Lassi",
    "Chicken (Chili Chipotle) Salad",
    "Chicken (Crispy Peri Peri) Salad",
    "Mini Chicken (Crispy Peri Peri) Salad",
    "Veg",
    "Orange - Juice",
    "Mixed Fruit Custard",
    "Papdi Chaat",
    "Bhel Puri",
    "Paneer (Mexican) Pro-Salad",
    "Veg Cheese Sandwich",
    "Roti Veg Curry Combo",
    "Honey Nut Crunch Ice cream",
    "Fresh Tender Coconut Water",
    "Cheese Infused Garlic Bread",
    "Cold Coffee ( Please return the bottle).",
    "Matar Kulcha",
    "Mushroom (Crispy) Burrito",
    "Potato (Peri Peri) Salad",
    "Chicken (BBQ) Quesadilla",
    "Potato (Peri Peri) Burrito",
    "Chilli Bean Patty 6 Inch.",
    "1 Fajita Veg Taco",
    "Egg Tossed Salad",
    "Sugarcane Juice (200 ml)",
    "Chicken",
    "Make Your Own Salad Non Veg",
    "Roti Non Veg Curry Combo",
    "Extra Protein",
    "Extra Chicken (BBQ)",
    "Chicken (BBQ) Pro-Bowl",
    "Chicken Tossed Salad",
    "Dahi Samosa Chaat",
    "Chicken Himalayan Momo",
    "Cut Fruit",
    "Chicken Roll",
    "Veg Starter",
    "Spanish Delight",
    "Fajita Veg Bowl",
    "Rice Non Veg Curry Combo",
    "Belgian Chocolate",
    "Roasted Californian Almond Ice cream",
    "Splish Splash Ice cream",
    "Apple Shake (200 ML)",
    "Chocolate Milk Shake  300 Ml",
    "Mushroom (Crispy) Salad",
    "Mini Paneer (BBQ) Bowl",
    "Packed Products",
    "Faluda",
    "Veg Salad",
    "Mini Fajita Veg Bowl",
    "Potato (Peri Peri) Bowl",
    "Mixed Fruit.",
    "Non Veg Salad",
    "Chicken (Crispy Peri Peri) - Side",
    "Butterscotch",
    "Mini Mushroom (Crispy) Burrito",
    "Paneer Capsicum Sandwich",
    "Chilli Chicken Frankie",
    "Mini Paneer (BBQ) Salad",
    "Tawa Chicken Frankie",
    "Extra Egg",
    "Fresh Sugar Cane juice",
    "Non-Veg Choix Bowl",
    "Badam Shake (200 ML)",
    "Spl Lassi",
    "Reboot Tea/Lemon Tea",
    "Delhi Wale Chatpate Chole With Rice",
    "Red Velvet",
    "Cotton Candy Ice cream",
    "Bavarian Chocolate Ice cream",
    "Corn Cheese Sandwich",
    "Chole Poori",
    "Mini Chicken (Crispy Peri Peri) Burrito",
    "Three cheers chocolate Ice cream",
    "Badam Milk",
    "Chocolate Danish",
    "Mississippi Mud Ice cream",
    "Cheesy Tostada",
    "Veg Cheese Frankie",
    "Chicken (Crispy Peri Peri) Quesadilla",
    "Water Melon Juice",
    "Paneer Tikka 6 Inch.",
    "Tandoori Chicken 6 Inch.",
    "Veg Butter Maggie",
    "Tuna 6 Inch.",
    "Mini Paneer (Mexican) Bowl",
    "Chicken Beetroot Wrap",
    "Paneer Puff",
    "Blueberry Blast Muffin",
    "Mini Fajita Veg Salad",
    "Premium Paan Flavour",
    "Boiled Egg",
    "Beetroot Juice",
    "Evening Snacks",
    "Iced Flat Mocha Reg",
]
                 


def find_similar_products(target_product, product_list):
    """
    Find products similar to the target product name.
    
    Args:
        target_product (str): The product name to search for
        product_list (list): List of all available products
    
    Returns:
        list: List of similar product names
    """
    
    def similarity_score(a, b):
        """Calculate similarity score between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def contains_word(target, product):
        """Check if target word is contained in product name"""
        target_words = re.findall(r'\w+', target.lower())
        product_words = re.findall(r'\w+', product.lower())
        
        for target_word in target_words:
            for product_word in product_words:
                if target_word in product_word or product_word in target_word:
                    return True
        return False
    
    def fuzzy_match(target, product, threshold=0.6):
        """Check if products are similar using fuzzy matching"""
        return similarity_score(target, product) >= threshold
    
    target_lower = target_product.lower().strip()
    similar_products = []
    
    for product in product_list:
        product_lower = product.lower().strip()
        
        # Skip if same product
        if product_lower == target_lower:
            similar_products.append(product)
            continue
            
        # Check for exact substring match
        if target_lower in product_lower or product_lower in target_lower:
            similar_products.append(product)
            continue
            
        # Check for word-level matches
        if contains_word(target_product, product):
            similar_products.append(product)
            continue
            
        # Check for fuzzy similarity
        if fuzzy_match(target_product, product):
            similar_products.append(product)
    
    # Sort by similarity score (descending)
    similar_products.sort(key=lambda x: similarity_score(target_product, x), reverse=True)
    
    return similar_products

# Test the function
def test_similarity_function(query, top_res):
        similar_items = find_similar_products(query, sample_products)[:top_res]
        print(f"\nSimilar items for '{query}' : {similar_items}:")
        similar_items = " , ".join(str(x) for x in similar_items)
        return similar_items


def regex_search(query, top_res):
        """
        search products present in product item list or not, if not return similar items
        """
        product_name = classify(query).strip().lower()
        if product_name == "no":
            return "true"
        else :
            if product_name.lower() in (s.lower() for s in sample_products) :
                return "true"
            else :
                return test_similarity_function(product_name, top_res=top_res)


# # Advanced version with configurable similarity threshold
# def find_similar_products_advanced(target_product, product_list, 
#                                  similarity_threshold=0.6, 
#                                  max_results=10):
#     """
#     Advanced version with configurable parameters
    
#     Args:
#         target_product (str): The product name to search for
#         product_list (list): List of all available products
#         similarity_threshold (float): Minimum similarity score (0.0 to 1.0)
#         max_results (int): Maximum number of results to return
    
#     Returns:
#         list: List of tuples (product_name, similarity_score)
#     """
#     import re
#     from difflib import SequenceMatcher
    
#     def similarity_score(a, b):
#         return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
#     def word_overlap_score(target, product):
#         """Calculate word overlap score"""
#         target_words = set(re.findall(r'\w+', target.lower()))
#         product_words = set(re.findall(r'\w+', product.lower()))
        
#         if not target_words or not product_words:
#             return 0.0
            
#         intersection = target_words.intersection(product_words)
#         union = target_words.union(product_words)
        
#         return len(intersection) / len(union)
    
#     results = []
#     target_lower = target_product.lower().strip()
    
#     for product in product_list:
#         product_lower = product.lower().strip()
        
#         # Calculate different similarity metrics
#         sequence_sim = similarity_score(target_product, product)
#         word_overlap = word_overlap_score(target_product, product)
        
#         # Combined score (weighted average)
#         combined_score = (sequence_sim * 0.7) + (word_overlap * 0.3)
        
#         # Check if product meets threshold
#         if combined_score >= similarity_threshold or target_lower in product_lower or product_lower in target_lower:
#             results.append((product, combined_score))
    
#     # Sort by score (descending) and limit results
#     results.sort(key=lambda x: x[1], reverse=True)
#     return results[:max_results]


# # Example usage of advanced function
# print("\n" + "="*50)
# print("ADVANCED FUNCTION EXAMPLE")
# print("="*50)

# advanced_results = find_similar_products_advanced("chai", sample_products, 
#                                                 similarity_threshold=0.3, 
#                                                 max_results=8)

# print(f"\nAdvanced results for 'chai' (with similarity scores):")
# for product, score in advanced_results:
#     print(f"  - {product} (score: {score:.3f})")