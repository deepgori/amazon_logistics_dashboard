import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import os

# --- Configuration Parameters ---
NUM_ORDERS = 100000  # Total number of orders to simulate
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
PRIME_MEMBER_RATIO = 0.70 # 70% of orders are from Prime members

# Delivery Time Assumptions (in days)
PRIME_DELIVERY_AVG_DAYS = 1.5
PRIME_DELIVERY_STD_DEV = 0.5
STANDARD_DELIVERY_AVG_DAYS = 6.0
STANDARD_DELIVERY_STD_DEV = 1.5

# Delay Probabilities
PRIME_DELAY_PROBABILITY = 0.05
STANDARD_DELAY_PROBABILITY = 0.20
AVERAGE_DELAY_DAYS = 2

# Carrier Distribution
PRIME_CARRIER_DIST = {
    'AMZL': 0.85, # Amazon Logistics
    'UPS': 0.07,
    'USPS': 0.05,
    'FedEx': 0.03
}
STANDARD_CARRIER_DIST = {
    'AMZL': 0.20,
    'UPS': 0.40,
    'USPS': 0.30,
    'FedEx': 0.10
}

# Cost Assumptions (conceptual, for Amazon)
BASE_AMZL_COST_PER_PACKAGE = 5.00
BASE_3PC_COST_PER_PACKAGE = 4.00
PRIME_EXPEDITED_COST_PREMIUM = 1.2

# --- Geographic Enrichment Configuration ---
US_ZIP_CODES_CSV = 'data/us_zip_codes.csv'
REAL_ZIP_CODES = [] # This will store actual ZIP codes from the CSV

# --- Output File Configuration ---
SIMULATED_ORDERS_CSV = 'data/simulated_orders.csv' # <-- THIS LINE WAS THE CAUSE OF NameError

# --- Initialize Faker ---
fake = Faker('en_US')

# --- Helper Functions ---
def get_random_date(start, end):
    """Generates a random datetime within a given range."""
    return start + timedelta(days=random.randint(0, (end - start).days))

def get_carrier(is_prime):
    """Selects a carrier based on Prime status and defined distributions."""
    if is_prime:
        return random.choices(list(PRIME_CARRIER_DIST.keys()), weights=list(PRIME_CARRIER_DIST.values()), k=1)[0]
    else:
        return random.choices(list(STANDARD_CARRIER_DIST.keys()), weights=list(STANDARD_CARRIER_DIST.values()), k=1)[0]

def get_delivery_time(is_prime):
    """Calculates a randomized delivery time in days."""
    if is_prime:
        return max(1, round(np.random.normal(PRIME_DELIVERY_AVG_DAYS, PRIME_DELIVERY_STD_DEV)))
    else:
        return max(1, round(np.random.normal(STANDARD_DELIVERY_AVG_DAYS, STANDARD_DELIVERY_STD_DEV)))

# --- Function to load real ZIP codes ---
def load_real_zip_codes(filepath):
    """Loads and returns a list of unique, 5-digit string ZIP codes from a CSV."""
    try:
        df_zip = pd.read_csv(filepath)
        # Assuming the column is named 'zip' and padding it to 5 digits
        if 'zip' in df_zip.columns:
            real_zips = df_zip['zip'].astype(str).str.zfill(5).tolist()
            real_zips = [z for z in real_zips if len(z) == 5 and z.isdigit()] # Filter for valid 5-digit strings
            return list(set(real_zips)) # Return unique zips
        elif 'ZIP' in df_zip.columns: # Alternative common name
            real_zips = df_zip['ZIP'].astype(str).str.zfill(5).tolist()
            real_zips = [z for z in real_zips if len(z) == 5 and z.isdigit()]
            return list(set(real_zips))
        else:
            print(f"Error: No 'zip' or 'ZIP' column found in {filepath}. Cannot load real ZIP codes.")
            return []
    except FileNotFoundError:
        print(f"Error: Zip code file not found at {filepath}. Please ensure it exists.")
        return []
    except Exception as e:
        print(f"Error loading zip code file: {e}")
        return []

# --- Main Data Generation Logic ---
print("Starting data generation...")

# Load real ZIP codes at the beginning of the script
REAL_ZIP_CODES = load_real_zip_codes(US_ZIP_CODES_CSV)
if not REAL_ZIP_CODES:
    print("No real ZIP codes loaded. Proceeding with Faker's random postcodes (may lead to unmapped zips).")
    use_faker_postcode = True
else:
    print(f"Loaded {len(REAL_ZIP_CODES)} unique real ZIP codes for simulation.")
    use_faker_postcode = False

orders_data = []

for i in range(NUM_ORDERS):
    order_id = f"ORD-{i:07d}"
    customer_id = f"CUST-{random.randint(10000, 99999):05d}"
    order_date = get_random_date(START_DATE, END_DATE)
    is_prime_member = random.random() < PRIME_MEMBER_RATIO

    # Determine expected and actual delivery dates
    delivery_days = get_delivery_time(is_prime_member)
    expected_delivery_date = order_date + timedelta(days=delivery_days)
    actual_delivery_date = expected_delivery_date

    # Simulate delays
    if (is_prime_member and random.random() < PRIME_DELAY_PROBABILITY) or \
       (not is_prime_member and random.random() < STANDARD_DELAY_PROBABILITY):
        actual_delivery_date += timedelta(days=random.randint(1, AVERAGE_DELAY_DAYS))

    # Determine delivery status
    if actual_delivery_date < expected_delivery_date:
        delivery_status = 'Early'
    elif actual_delivery_date == expected_delivery_date:
        delivery_status = 'On-Time'
    else:
        delivery_status = 'Late'

    # Assign carrier
    carrier = get_carrier(is_prime_member)

    # Calculate conceptual cost to Amazon
    base_cost = BASE_AMZL_COST_PER_PACKAGE if carrier == 'AMZL' else BASE_3PC_COST_PER_PACKAGE
    delivery_cost_to_amazon = base_cost
    if is_prime_member:
        delivery_cost_to_amazon *= PRIME_EXPEDITED_COST_PREMIUM
    delivery_cost_to_amazon = round(delivery_cost_to_amazon * random.uniform(0.9, 1.1), 2)

    # --- Generate destination ZIP code using REAL ZIP codes ---
    if not use_faker_postcode:
        # Ensure REAL_ZIP_CODES is not empty before attempting random.choice
        if REAL_ZIP_CODES:
            destination_zip_code = random.choice(REAL_ZIP_CODES)
        else: # Fallback to faker if REAL_ZIP_CODES unexpectedly becomes empty despite initial check
            destination_zip_code = fake.postcode()
    else:
        destination_zip_code = fake.postcode()


    orders_data.append({
        'order_id': order_id,
        'customer_id': customer_id,
        'order_date': order_date.strftime('%Y-%m-%d'),
        'is_prime_member': is_prime_member,
        'expected_delivery_date': expected_delivery_date.strftime('%Y-%m-%d'),
        'actual_delivery_date': actual_delivery_date.strftime('%Y-%m-%d'),
        'delivery_status': delivery_status,
        'carrier': carrier,
        'delivery_cost_to_amazon': delivery_cost_to_amazon,
        'product_id': f"PROD-{random.randint(100, 999):03d}",
        'order_quantity': random.randint(1, 5),
        'destination_zip_code': destination_zip_code
    })

# Create DataFrame
df_orders = pd.DataFrame(orders_data)

# --- Save to CSV ---
output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Use SIMULATED_ORDERS_CSV variable for output file path
output_file = os.path.join(output_dir, SIMULATED_ORDERS_CSV.split('/')[-1])
df_orders.to_csv(output_file, index=False)

print(f"\nData generation complete. {NUM_ORDERS} orders saved to '{output_file}'")
print("\nFirst 5 rows of the generated data:")
print(df_orders.head())
print("\nDescriptive statistics for Prime vs Standard delivery days:")
df_orders['delivery_days_actual'] = (pd.to_datetime(df_orders['actual_delivery_date']) - pd.to_datetime(df_orders['order_date'])).dt.days
print(df_orders.groupby('is_prime_member')['delivery_days_actual'].describe())

print("\nCarrier distribution for Prime members:")
print(df_orders[df_orders['is_prime_member'] == True]['carrier'].value_counts(normalize=True))

print("\nCarrier distribution for Standard members:")
print(df_orders[df_orders['is_prime_member'] == False]['carrier'].value_counts(normalize=True))