import pandas as pd
import numpy as np
import math
from olist.data import Olist
from olist.order import Order


class Review:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_review_length(self):
        """
        Returns a DataFrame with:
       'review_id', 'length_review', 'review_score'
        """
        reviews = self.data['order_reviews'].copy()

        # Length of the review comment message (0 if the review is empty/NaN)
        reviews['length_review'] = reviews['review_comment_message'] \
            .fillna('') \
            .map(len)

        return reviews[['review_id', 'length_review', 'review_score']]

    def get_main_product_category(self):
        """
        Returns a DataFrame with:
       'review_id', 'order_id','product_category_name'
        """
        # Link each order to its products
        order_items = self.data['order_items'][['order_id', 'product_id']]
        products = self.data['products'][['product_id', 'product_category_name']]

        order_products = order_items.merge(products, on='product_id')

        # For each order, keep the most frequent product category (the "main" one)
        main_cat = order_products \
            .groupby('order_id')['product_category_name'] \
            .agg(lambda x: x.value_counts().index[0]) \
            .reset_index()

        # Attach the review_id of each order
        reviews = self.data['order_reviews'][['review_id', 'order_id']]
        df = reviews.merge(main_cat, on='order_id')

        return df[['review_id', 'order_id', 'product_category_name']]

    def get_training_data(self):
        """
        Returns a DataFrame with:
       'review_id', 'length_review', 'review_score',
       'order_id', 'product_category_name'
        """
        training_set = self.get_review_length() \
            .merge(self.get_main_product_category(), on='review_id')

        return training_set