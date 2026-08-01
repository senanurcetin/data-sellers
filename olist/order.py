import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """
        # Make a copy before using inplace=True so as to avoid modifying self.data
        orders = self.data['orders'].copy()

        # Filter delivered orders
        if is_delivered:
            orders = orders.query("order_status == 'delivered'").copy()

        # Convert date columns to datetime
        orders['order_purchase_timestamp'] = pd.to_datetime(
            orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date'] = pd.to_datetime(
            orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] = pd.to_datetime(
            orders['order_estimated_delivery_date'])

        # Compute wait time (in days)
        orders['wait_time'] = (
            orders['order_delivered_customer_date'] -
            orders['order_purchase_timestamp']
        ) / np.timedelta64(24, 'h')

        # Compute expected wait time (in days)
        orders['expected_wait_time'] = (
            orders['order_estimated_delivery_date'] -
            orders['order_purchase_timestamp']
        ) / np.timedelta64(24, 'h')

        # Compute delay vs expected (only positive delays, 0 if delivered on time)
        def compute_delay(row):
            delay = (row['order_delivered_customer_date'] -
                     row['order_estimated_delivery_date']) / np.timedelta64(24, 'h')
            return delay if delay > 0 else 0

        orders['delay_vs_expected'] = orders.apply(compute_delay, axis=1)

        return orders[[
            'order_id', 'wait_time', 'expected_wait_time',
            'delay_vs_expected', 'order_status'
        ]]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()

        reviews['dim_is_five_star'] = reviews['review_score'].apply(
            lambda x: 1 if x == 5 else 0)
        reviews['dim_is_one_star'] = reviews['review_score'].apply(
            lambda x: 1 if x == 1 else 0)

        return reviews[[
            'order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score'
        ]]

    def get_number_items(self):
        """
        Returns a DataFrame with:
        order_id, number_of_items
        """
        items = self.data['order_items'].copy()

        number_items = items.groupby('order_id', as_index=False).agg(
            number_of_items=('order_item_id', 'count'))

        return number_items[['order_id', 'number_of_items']]

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        items = self.data['order_items'].copy()

        number_sellers = items.groupby('order_id', as_index=False).agg(
            number_of_sellers=('seller_id', 'nunique'))

        return number_sellers[['order_id', 'number_of_sellers']]

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        items = self.data['order_items'].copy()

        price_freight = items.groupby('order_id', as_index=False).agg(
            price=('price', 'sum'),
            freight_value=('freight_value', 'sum'))

        return price_freight[['order_id', 'price', 'freight_value']]

    # Optional
    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        """
        # Copy the relevant tables
        orders = self.data['orders'].copy()
        order_items = self.data['order_items'].copy()
        sellers = self.data['sellers'].copy()
        customers = self.data['customers'].copy()
        geo = self.data['geolocation'].copy()

        # Since one zip code can have multiple lat/lng, take the first per zip
        geo = geo.groupby('geolocation_zip_code_prefix', as_index=False).agg(
            {'geolocation_lat': 'first', 'geolocation_lng': 'first'})

        # Merge sellers with their geolocation
        sellers_geo = sellers.merge(
            geo,
            left_on='seller_zip_code_prefix',
            right_on='geolocation_zip_code_prefix'
        )[['seller_id', 'seller_zip_code_prefix',
           'geolocation_lat', 'geolocation_lng']]
        sellers_geo = sellers_geo.rename(columns={
            'geolocation_lat': 'seller_lat',
            'geolocation_lng': 'seller_lng'})

        # Merge customers with their geolocation
        customers_geo = customers.merge(
            geo,
            left_on='customer_zip_code_prefix',
            right_on='geolocation_zip_code_prefix'
        )[['customer_id', 'customer_zip_code_prefix',
           'geolocation_lat', 'geolocation_lng']]
        customers_geo = customers_geo.rename(columns={
            'geolocation_lat': 'customer_lat',
            'geolocation_lng': 'customer_lng'})

        # Match customers to orders
        orders_customers = orders[['order_id', 'customer_id']].merge(
            customers_geo, on='customer_id')

        # Match sellers to orders (via order_items)
        orders_sellers = order_items[['order_id', 'seller_id']].merge(
            sellers_geo, on='seller_id')

        # Bring both together
        matching = orders_sellers.merge(orders_customers, on='order_id')

        # Remove any rows with missing coordinates
        matching = matching.dropna()

        # Compute the haversine distance for each row
        matching['distance_seller_customer'] = matching.apply(
            lambda row: haversine_distance(
                row['seller_lng'], row['seller_lat'],
                row['customer_lng'], row['customer_lat']),
            axis=1)

        # An order can have several sellers -> average the distances per order
        distance = matching.groupby('order_id', as_index=False).agg(
            distance_seller_customer=('distance_seller_customer', 'mean'))

        return distance[['order_id', 'distance_seller_customer']]

    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_items', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """
        # Merge all the feature DataFrames on order_id
        training_set = self.get_wait_time(is_delivered) \
            .merge(self.get_review_score(), on='order_id') \
            .merge(self.get_number_items(), on='order_id') \
            .merge(self.get_number_sellers(), on='order_id') \
            .merge(self.get_price_and_freight(), on='order_id')

        # Optionally add distance
        if with_distance_seller_customer:
            training_set = training_set.merge(
                self.get_distance_seller_customer(), on='order_id')

        # Drop rows with missing values
        return training_set.dropna()