def user_login(username: str, password: str) -> dict:  
    """  
    Logs in a user.  
    Input: username (str), password (str)  
    Output: dict with user details or error message  
    """  
    return {"status": "success", "user_id": 123}  
  
def user_register(username: str, email: str, password: str) -> dict:  
    """  
    Registers a new user.  
    Input: username (str), email (str), password (str)  
    Output: dict with registration status  
    """  
    return {"status": "success", "user_id": 123}  
  
def reset_password(email: str) -> dict:  
    """  
    Sends a password reset link to the user's email.  
    Input: email (str)  
    Output: dict with reset status  
    """  
    return {"status": "email_sent"}  

def fetch_product_details(product_id: int) -> dict:  
    """  
    Fetches details of a product.  
    Input: product_id (int)  
    Output: dict with product details  
    """  
    return {"product_id": product_id, "name": "Sample Product", "price": 100.0}  
  
def search_products(query: str, category: str = None) -> list:  
    """  
    Searches for products based on a query and optional category.  
    Input: query (str), category (str, optional)  
    Output: list of matching products  
    """  
    return [{"product_id": 1, "name": "Sample Product", "price": 100.0}]  

def check_stock(product_id: int) -> dict:  
    """  
    Checks stock availability for a product.  
    Input: product_id (int)  
    Output: dict with stock status  
    """  
    return {"product_id": product_id, "stock": 50}  
  
def update_inventory(product_id: int, quantity: int) -> dict:  
    """  
    Updates inventory for a product.  
    Input: product_id (int), quantity (int)  
    Output: dict with update status  
    """  
    return {"status": "success", "product_id": product_id, "new_stock": 50 - quantity}  

def add_to_cart(user_id: int, product_id: int, quantity: int) -> dict:  
    """  
    Adds a product to the user's shopping cart.  
    Input: user_id (int), product_id (int), quantity (int)  
    Output: dict with cart status  
    """  
    return {"status": "success", "cart_id": 1}  
  
def calculate_checkout_total(cart_id: int) -> dict:  
    """  
    Calculates the total cost of items in the cart.  
    Input: cart_id (int)  
    Output: dict with total cost  
    """  
    return {"cart_id": cart_id, "total": 150.0}  

def process_payment(order_id: int, payment_method: str) -> dict:  
    """  
    Processes payment for an order.  
    Input: order_id (int), payment_method (str)  
    Output: dict with payment status  
    """  
    return {"status": "success", "transaction_id": "TX12345"}  
  
def refund_payment(transaction_id: str) -> dict:  
    """  
    Processes a refund for a transaction.  
    Input: transaction_id (str)  
    Output: dict with refund status  
    """  
    return {"status": "success", "refund_id": "RF123"}

def calculate_shipping_cost(order_id: int, shipping_method: str) -> dict:  
    """  
    Calculates the shipping cost for an order.  
    Input: order_id (int), shipping_method (str)  
    Output: dict with shipping cost  
    """  
    return {"order_id": order_id, "shipping_method": shipping_method, "cost": 10.0}  
  
def generate_shipping_label(order_id: int) -> dict:  
    """  
    Generates a shipping label for an order.  
    Input: order_id (int)  
    Output: dict with shipping label details  
    """  
    return {"order_id": order_id, "label_url": "https://example.com/shipping_label.pdf"}  
  
def track_shipment(tracking_number: str) -> dict:  
    """  
    Tracks the shipment status using a tracking number.  
    Input: tracking_number (str)  
    Output: dict with shipment status  
    """  
    return {"tracking_number": tracking_number, "status": "In Transit", "estimated_delivery": "2023-10-15"}  

def create_order(user_id: int, cart_id: int) -> dict:  
    """  
    Creates an order from the user's cart.  
    Input: user_id (int), cart_id (int)  
    Output: dict with order details  
    """  
    return {"order_id": 101, "status": "created", "total": 150.0}  
  
def get_order_status(order_id: int) -> dict:  
    """  
    Retrieves the status of an order.  
    Input: order_id (int)  
    Output: dict with order status  
    """
    return {"order_id" : order_id, "status": "shipped", "estimated_delivery": "2023-10-15"}

def cancel_order(order_id: int) -> dict:  
    """  
    Cancels an order if it is not yet shipped.  
    Input: order_id (int)  
    Output: dict with cancellation status  
    """  
    return {"order_id": order_id, "status": "cancelled"}  
  
def generate_invoice(order_id: int) -> dict:  
    """  
    Generates an invoice for an order.  
    Input: order_id (int)  
    Output: dict with invoice details  
    """  
    return {"order_id": order_id, "invoice_url": "https://example.com/invoice_101.pdf"}  

def create_promo_code(code: str, discount: float, expiry_date: str) -> dict:  
    """  
    Creates a new promo code.  
    Input: code (str), discount (float), expiry_date (str in YYYY-MM-DD format)  
    Output: dict with promo code creation status  
    """  
    return {"promo_code": code, "discount": discount, "expiry_date": expiry_date, "status": "created"}  
  
def apply_promo_code(cart_id: int, promo_code: str) -> dict:  
    """  
    Applies a promo code to a shopping cart.  
    Input: cart_id (int), promo_code (str)  
    Output: dict with updated cart total and discount applied  
    """  
    return {"cart_id": cart_id, "promo_code": promo_code, "discount_applied": 10.0, "new_total": 140.0}  
  
def send_marketing_email(user_id: int, subject: str, content: str) -> dict:  
    """  
    Sends a marketing email to a user.  
    Input: user_id (int), subject (str), content (str)  
    Output: dict with email sending status  
    """  
    return {"user_id": user_id, "status": "email_sent", "subject": subject}  
  
def track_promo_usage(promo_code: str) -> dict:  
    """  
    Tracks the usage of a promo code.  
    Input: promo_code (str)  
    Output: dict with usage statistics  
    """  
    return {"promo_code": promo_code, "times_used": 25, "remaining_uses": 75}  