def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars', 'review_score', 'cost_of_reviews'
        """
        orders_reviews = self.order.get_review_score()
        orders_sellers = self.data['order_items'][['order_id', 'seller_id']] \
            .drop_duplicates()

        df = orders_sellers.merge(orders_reviews, on='order_id')

        # Cost incurred by Olist for each review score (bad reviews are costly)
        df['cost_of_reviews'] = df.review_score.map({
            1: 100,
            2: 50,
            3: 40,
            4: 0,
            5: 0
        })

        df = df.groupby('seller_id', as_index=False).agg({
            'dim_is_one_star': 'mean',
            'dim_is_five_star': 'mean',
            'review_score': 'mean',
            'cost_of_reviews': 'sum'
        })

        df.columns = [
            'seller_id', 'share_of_one_stars', 'share_of_five_stars',
            'review_score', 'cost_of_reviews'
        ]

        return df

    def get_training_data(self):
        """
        Returns a DataFrame with:
        ['seller_id', 'seller_city', 'seller_state', 'delay_to_carrier',
        'wait_time', 'date_first_sale', 'date_last_sale', 'months_on_olist',
        'share_of_one_stars', 'share_of_five_stars', 'review_score',
        'cost_of_reviews', 'n_orders', 'quantity', 'quantity_per_order',
        'sales', 'revenues', 'profits']
        """
        training_set = \
            self.get_seller_features() \
                .merge(
                    self.get_seller_delay_wait_time(), on='seller_id'
                ).merge(
                    self.get_active_dates(), on='seller_id'
                ).merge(
                    self.get_review_score(), on='seller_id'
                ).merge(
                    self.get_quantity(), on='seller_id'
                ).merge(
                    self.get_sales(), on='seller_id'
                )

        # Compute the economics (revenues, profits)
        # Olist takes a 10% cut on the product price of each order + charges a
        # 80 BRL monthly fee per seller
        olist_monthly_fee = 80
        olist_sales_cut = 0.1

        training_set['revenues'] = \
            olist_monthly_fee * training_set['months_on_olist'] \
            + olist_sales_cut * training_set['sales']

        training_set['profits'] = \
            training_set['revenues'] - training_set['cost_of_reviews']

        return training_set