import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np

orders = pd.read_csv('archive/olist_orders_dataset.csv')
items = pd.read_csv('archive/olist_order_items_dataset.csv')
payments = pd.read_csv('archive/olist_order_payments_dataset.csv')
reviews = pd.read_csv('archive/olist_order_reviews_dataset.csv')
customers = pd.read_csv('archive/olist_customers_dataset.csv')
products = pd.read_csv('archive/olist_products_dataset.csv')
sellers = pd.read_csv('archive/olist_sellers_dataset.csv')
translation = pd.read_csv('archive/product_category_name_translation.csv')

products = products.merge(translation, on='product_category_name', how='left')

conn = sqlite3.connect('olist.db')

orders.to_sql('orders', conn, if_exists='replace', index=False)
items.to_sql('items', conn, if_exists='replace', index=False)
payments.to_sql('payments', conn, if_exists='replace', index=False)
reviews.to_sql('reviews', conn, if_exists='replace', index=False)
customers.to_sql('customers', conn, if_exists='replace', index=False)
products.to_sql('products', conn, if_exists='replace', index=False)
sellers.to_sql('sellers', conn, if_exists='replace', index=False)

print(pd.read_sql("""
SELECT 
    COUNT(DISTINCT o.order_id) as orders,
    COUNT(DISTINCT o.customer_id) as customers,
    ROUND(SUM(p.payment_value), 2) as revenue
FROM orders o
JOIN payments p ON o.order_id = p.order_id
""", conn))

print(pd.read_sql("""
SELECT order_status, COUNT(*) as cnt,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders), 2) as pct
FROM orders
GROUP BY 1 ORDER BY 2 DESC
""", conn))

print(pd.read_sql("""
SELECT payment_type,
       COUNT(*) as cnt,
       ROUND(AVG(payment_value), 2) as avg_value,
       ROUND(SUM(payment_value), 2) as total
FROM payments
GROUP BY 1 ORDER BY total DESC
""", conn))

print(pd.read_sql("""
SELECT p.product_category_name_english as category,
       ROUND(SUM(i.price), 2) as revenue,
       COUNT(*) as items_sold
FROM items i
JOIN products p ON i.product_id = p.product_id
GROUP BY 1
ORDER BY revenue DESC
LIMIT 10
""", conn))

print(pd.read_sql("""
SELECT 
    ROUND(AVG(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)), 1) as avg_days,
    ROUND(MIN(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)), 1) as min_days,
    ROUND(MAX(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)), 1) as max_days
FROM orders
WHERE order_delivered_customer_date IS NOT NULL
""", conn))

print(pd.read_sql("""
SELECT 
    CASE 
        WHEN days <= 7 THEN '0-7 days'
        WHEN days <= 14 THEN '8-14 days'
        WHEN days <= 30 THEN '15-30 days'
        ELSE '30+ days'
    END as group_days,
    ROUND(AVG(review_score), 2) as avg_score,
    COUNT(*) as cnt
FROM (
    SELECT julianday(o.order_delivered_customer_date) - julianday(o.order_purchase_timestamp) as days,
           r.review_score
    FROM orders o
    JOIN reviews r ON o.order_id = r.order_id
    WHERE o.order_delivered_customer_date IS NOT NULL
)
GROUP BY 1
""", conn))

print(pd.read_sql("""
SELECT s.seller_id, s.seller_city, s.seller_state,
       ROUND(SUM(i.price), 2) as revenue,
       COUNT(DISTINCT i.order_id) as orders
FROM items i
JOIN sellers s ON i.seller_id = s.seller_id
GROUP BY 1
ORDER BY revenue DESC
LIMIT 10
""", conn))

print(pd.read_sql("""
SELECT c.customer_state,
       COUNT(DISTINCT o.order_id) as orders,
       ROUND(SUM(p.payment_value), 2) as revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN payments p ON o.order_id = p.order_id
GROUP BY 1
ORDER BY revenue DESC
LIMIT 10
""", conn))

orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
max_date = orders['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

rfm = orders.merge(payments.groupby('order_id')['payment_value'].sum().reset_index(), on='order_id')
rfm = rfm.groupby('customer_id').agg({
    'order_purchase_timestamp': lambda x: (max_date - x.max()).days,
    'order_id': 'count',
    'payment_value': 'sum'
}).reset_index()

rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
rfm['R'] = pd.qcut(rfm['recency'], 4, labels=[4,3,2,1])
rfm['F'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M'] = pd.qcut(rfm['monetary'], 4, labels=[1,2,3,4])
rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

print(rfm['RFM_Score'].value_counts().head(10))

orders['order_month'] = orders['order_purchase_timestamp'].dt.to_period('M')
orders['cohort'] = orders.groupby('customer_id')['order_purchase_timestamp'].transform('min').dt.to_period('M')

cohort = orders.groupby(['cohort', 'order_month']).agg(n_customers=('customer_id', 'nunique')).reset_index()
cohort['period'] = (cohort['order_month'] - cohort['cohort']).apply(lambda x: x.n)

cohort_pivot = cohort.pivot_table(index='cohort', columns='period', values='n_customers')
retention = cohort_pivot.divide(cohort_pivot[0], axis=0)

print(retention.iloc[:8, :6].round(3))

plt.figure(figsize=(10, 6))
orders['order_status'].value_counts().plot.bar(color='steelblue')
plt.title('Order Statuses')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
reviews['review_score'].value_counts().sort_index().plot.bar(color='salmon')
plt.title('Review Scores')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
orders['delivery_days'] = (pd.to_datetime(orders['order_delivered_customer_date']) - orders['order_purchase_timestamp']).dt.days
orders['delivery_days'].dropna().clip(0, 50).hist(bins=25, color='seagreen')
plt.title('Delivery Time (days)')
plt.xlabel('Days')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
data = retention.iloc[:10, :8].fillna(0).values
plt.imshow(data, cmap='YlGnBu', aspect='auto')
plt.colorbar()
plt.xticks(range(data.shape[1]), range(data.shape[1]))
plt.yticks(range(data.shape[0]), retention.index[:10].astype(str))
plt.title('Cohort Retention')
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        plt.text(j, i, f'{data[i, j]:.0%}', ha='center', va='center', color='black', fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
rfm['monetary'].clip(0, 2000).hist(bins=30, color='orange')
plt.title('Monetary Distribution (RFM)')
plt.xlabel('Monetary Value')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
top_states = pd.read_sql("""
SELECT customer_state, ROUND(SUM(payment_value),0) as revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN payments p ON o.order_id = p.order_id
GROUP BY 1 ORDER BY 2 DESC LIMIT 8
""", conn)
plt.barh(top_states['customer_state'], top_states['revenue'], color='mediumpurple')
plt.title('Top States by Revenue')
plt.xlabel('Revenue')
plt.tight_layout()
plt.show()