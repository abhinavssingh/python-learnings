
# Project Requirements Document
## Analyzing Customer Orders Using Python

### 1. Overview
This project focuses on analyzing customer orders for an e‑commerce company using Python.

### 2. Objectives
- Process and store customer order data efficiently
- Classify customers based on spending behavior
- Analyze product categories and purchasing patterns
- Generate business insights

### 3. Dependencies

`pip install -r requirements.txt`

### 4. Functional Requirements

#### 4.1 Store Customer Orders
- Create a list of customer names
- Store each order as tuples: (customer_name, product, price, category)
- Dictionary mapping customer name to list of products

#### 4.2 Product Classification
- Dictionary mapping product → category
- Set of unique categories
- Display all categories

#### 4.3 Customer Purchase Analysis
- Compute total spending per customer
- Classify customers: high (>100), moderate (50-100), low (<50)

#### 4.4 Business Insights Generation
- Revenue per category
- Unique products using set
- List customers who purchased electronics
- Top 3 highest-spending customers

#### 4.5 Organizing and Displaying Data
- Print summary per customer
- Set operations to find multi-category customers
- Identify customers who bought both electronics and clothing

### 5. Technical Requirements
- Python language
- Use lists, tuples, dictionaries, sets
- Use loops, sorting, list comprehensions

### 6. Deliverables
- Python scripts and Jyputer Notebook files
- Final written report with insights
