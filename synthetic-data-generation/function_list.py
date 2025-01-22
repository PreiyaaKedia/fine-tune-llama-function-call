function_list = [
    {  
    "type": "function",  
    "function": {  
        "name": "create_promo_code",  
        "description": "Creates a new promo code for discounts.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "code": {  
                    "type": "string",  
                    "description": "The promo code to be created."  
                },  
                "discount": {  
                    "type": "number",  
                    "description": "The discount percentage or amount for the promo code."  
                },  
                "expiry_date": {  
                    "type": "string",  
                    "description": "The expiration date of the promo code in YYYY-MM-DD format."  
                }  
            },  
            "required": ["code", "discount", "expiry_date"]  
        }  
    }  
},

{  
    "type": "function",  
    "function": {  
        "name": "apply_promo_code",  
        "description": "Applies a promo code to a shopping cart.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "cart_id": {  
                    "type": "integer",  
                    "description": "The ID of the shopping cart."  
                },  
                "promo_code": {  
                    "type": "string",  
                    "description": "The promo code to be applied."  
                }  
            },  
            "required": ["cart_id", "promo_code"]  
        }  
    }  
}  ,

{  
    "type": "function",  
    "function": {  
        "name": "track_promo_usage",  
        "description": "Tracks the usage statistics of a promo code.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "promo_code": {  
                    "type": "string",  
                    "description": "The promo code to track."  
                }  
            },  
            "required": ["promo_code"]  
        }  
    }  
}  ,

{  
    "type": "function",  
    "function": {  
        "name": "calculate_checkout_total",  
        "description": "Calculates the total cost of items in the cart after applying discounts.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "cart_id": {  
                    "type": "integer",  
                    "description": "The ID of the shopping cart."  
                }  
            },  
            "required": ["cart_id"]  
        }  
    }  
}  ,

{  
    "type": "function",  
    "function": {  
        "name": "create_order",  
        "description": "Creates an order from the user's shopping cart.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "user_id": {  
                    "type": "integer",  
                    "description": "The ID of the user placing the order."  
                },  
                "cart_id": {  
                    "type": "integer",  
                    "description": "The ID of the shopping cart to create the order from."  
                }  
            },  
            "required": ["user_id", "cart_id"]  
        }  
    }  
}  ,

{  
    "type": "function",  
    "function": {  
        "name": "check_stock",  
        "description": "Checks the stock availability for a product.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "product_id": {  
                    "type": "integer",  
                    "description": "The ID of the product to check stock for."  
                }  
            },  
            "required": ["product_id"]  
        }  
    }  
}  ,

{  
    "type": "function",  
    "function": {  
        "name": "update_inventory",  
        "description": "Updates the inventory for a product by adjusting the stock quantity.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "product_id": {  
                    "type": "integer",  
                    "description": "The ID of the product to update."  
                },  
                "quantity": {  
                    "type": "integer",  
                    "description": "The quantity to adjust the stock by (positive or negative)."  
                }  
            },  
            "required": ["product_id", "quantity"]  
        }  
    }  
}  ,

{  
    "type": "function",  
    "function": {  
        "name": "fetch_product_details",  
        "description": "Fetches details of a specific product.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "product_id": {  
                    "type": "integer",  
                    "description": "The ID of the product to fetch details for."  
                }  
            },  
            "required": ["product_id"]  
        }  
    }  
}  ,

{  
    "type": "function",  
    "function": {  
        "name": "track_shipment",  
        "description": "Tracks the shipment status using a tracking number.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "tracking_number": {  
                    "type": "string",  
                    "description": "The tracking number of the shipment."  
                }  
            },  
            "required": ["tracking_number"]  
        }  
    }  
}  ,

{  
    "type": "function",  
    "function": {  
        "name": "get_order_status",  
        "description": "Retrieves the current status of an order.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "order_id": {  
                    "type": "integer",  
                    "description": "The ID of the order to check the status for."  
                }  
            },  
            "required": ["order_id"]  
        }  
    }  
}  ,

{  
    "type": "function",  
    "function": {  
        "name": "generate_invoice",  
        "description": "Generates an invoice for a specific order.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "order_id": {  
                    "type": "integer",  
                    "description": "The ID of the order to generate an invoice for."  
                }  
            },  
            "required": ["order_id"]  
        }  
    }  
}  ,

{  
    "type": "function",  
    "function": {  
        "name": "calculate_shipping_cost",  
        "description": "Calculates the shipping cost for an order based on the shipping method.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "order_id": {  
                    "type": "integer",  
                    "description": "The ID of the order to calculate shipping for."  
                },  
                "shipping_method": {  
                    "type": "string",  
                    "description": "The shipping method to use (e.g., standard, express)."  
                }  
            },  
            "required": ["order_id", "shipping_method"]  
        }  
    }  
}  ,
{  
    "type": "function",  
    "function": {  
        "name": "reset_password",  
        "description": "Sends a password reset link to the user.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "email": {  
                    "type": "string",  
                    "description": "The email id of the user requesting password reset."  
                },  
            },  
            "required": ["email"]  
        }  
    }  
}  ,
{  
    "type": "function",  
    "function": {  
        "name": "search_products",  
        "description": "Searches for products based on a query and optional category.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "query": {  
                    "type": "string",  
                    "description": "The search query to find matching products."  
                },  
                "category": {  
                    "type": "string",  
                    "description": "The category to filter products (optional).",
                }  
            },  
            "required": ["query"]  
        }  
    }  
} ,
{  
    "type": "function",  
    "function": {  
        "name": "add_to_cart",  
        "description": "Adds a product to the user's shopping cart.",  
        "parameters": {  
            "type": "object",  
            "properties": {  
                "user_id": {  
                    "type": "integer",  
                    "description": "The ID of the user adding the product to their cart."  
                },  
                "product_id": {  
                    "type": "integer",  
                    "description": "The ID of the product to be added to the cart."  
                },  
                "quantity": {  
                    "type": "integer",  
                    "description": "The quantity of the product to add to the cart."  
                }  
            },  
            "required": ["user_id", "product_id", "quantity"]  
        }  
    }  
} ,
]