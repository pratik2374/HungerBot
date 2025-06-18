from to_check_type import classify
import re
from difflib import SequenceMatcher

def load_list_from_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]
sample_products = load_list_from_txt("unique_products.txt")
print(f"Loaded {len(sample_products)} products from file.")
                 


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