# flake8: noqa
from langchain_core.prompts import PromptTemplate

#As a dbt expert, you are provided with descriptions of dbt models and their dependencies. Carefully analyze the content to answer the question below. Use only the information provided and avoid speculating beyond the given data. If an answer cannot be deduced , clearly state "I couldn't find the answer in the provided sources." When you provide an answer, you must cite all relevant sources that support your conclusions. See the example for how to structure your answer with sources.

template = """As a dbt expert, you are provided with descriptions of dbt models and their dependencies. Carefully analyze the content to answer the question below. Use only the information provided and avoid speculating beyond the given data. If a comprehensive answer cannot be deduced, clearly state "I couldn't find the answer in the provided sources." However, please summarize any relevant information that can be inferred from the content and state, "While a complete answer cannot be deduced, here's what is known from the provided sources:". Always include citations from the provided sources in your answer to support your conclusions. See the example for how to structure your answer with sources.



QUESTION: What are the sources of data for the customer_orders model and how is the data structured?
=========
Content: The customer_orders model, which aggregates order data to analyze customer purchasing behavior, includes columns for order_id, customer_id, order_date, and total_amount. It depends on the customer_details and order_items models for detailed data.
Source: model.project.customer_orders

Content: The customer_details model provides essential information about customers, containing columns such as customer_id, first_name, last_name, and email. This model is a dependency for the customer_orders model.
Source: model.project.customer_details

Content: The order_items model details each item within an order and includes columns such as order_id, product_id, quantity, and item_price. This model supports the total_amount calculation in the customer_orders model.
Source: model.project.order_items
=========
FINAL ANSWER:
The customer_orders model structures data to provide insights into customer purchasing behavior, featuring columns for order_id, customer_id, order_date, and total_amount. This model relies on detailed data sourced from the customer_details model for customer identification information and the order_items model for transaction specifics. Each model contributes distinct data that supports the analysis and reporting capabilities of the customer_orders model.

SOURCES:
- model.project.customer_orders
- model.project.customer_details
- model.project.order_items

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""




# template = """As a dbt expert, use the provided content, including main model and dependency model descriptions, to answer the question below. If the answer isn't available in the content, respond with "I couldn't find the answer in the provided sources." Always cite the sources of your information.

# QUESTION: What columns are included in the customer_orders model, and what are their sources?
# =========
# Main Model Description: The customer_orders model includes columns for order_id, customer_id, order_date, and total_amount. It aggregates order data to analyze customer purchasing behavior.
# Source: model.project.customer_orders

# Dependency 1 Description: The customer_details model contains customer_id, first_name, last_name, and email. It provides essential information about customers that is joined with orders in the customer_orders model.
# Source: model.project.customer_details

# Dependency 2 Description: The order_items model includes order_id, product_id, quantity, and item_price. It details each item within an order, contributing to the total_amount calculation in the customer_orders model.
# Source: model.project.order_items
# =========
# FINAL ANSWER:The customer_orders model includes the following columns:
# - order_id: Sourced from both customer_orders and order_items models.
# - customer_id: Sourced from the customer_details model.
# - order_date: Sourced from the customer_orders model.
# - total_amount: Calculated within the customer_orders model using data from the order_items model.

# SOURCES:
# - model.project.customer_orders
# - model.project.customer_details
# - model.project.order_items

# QUESTION: {question}
# =========
# {summaries}
# =========
# FINAL ANSWER:"""


# template = """As a dbt expert, use the provided content, including main model and dependency model descriptions, to answer the question below. If the answer isn't available in the content, respond with "I couldn't find the answer in the provided sources." Always cite the sources of your information.

# QUESTION: What types of data and columns are included in the customer_portal model?
# =========
# Content: The customer_portal model includes columns for customer_id, first_name, last_name, and subscription_type. It primarily stores data about customers' subscription details.
# Source: model_documentation-1
# Content: Additional metadata for the customer_portal model includes timestamps of account creation and last update.
# Source: model_documentation-2
# =========
# FINAL ANSWER: The customer_portal model includes columns for customer_id, first_name, last_name, subscription_type, account creation timestamp, and last update timestamp.
# SOURCES: model_documentation-1, model_documentation-2

# QUESTION: {question}
# =========
# {summaries}
# =========
# FINAL ANSWER:"""


# template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
# If you don't know the answer, just say that you don't know. Don't try to make up an answer.
# ALWAYS return a "SOURCES" part in your answer.

# QUESTION: Which state/country's law governs the interpretation of the contract?
# =========
# Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
# Source: 28-pl
# Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
# Source: 30-pl
# Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
# Source: 4-pl
# =========
# FINAL ANSWER: This Agreement is governed by English law.
# SOURCES: 28-pl

# QUESTION: What did the president say about Michael Jackson?
# =========
# Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
# Source: 0-pl
# Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
# Source: 24-pl
# Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
# Source: 5-pl
# Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
# Source: 34-pl
# =========
# FINAL ANSWER: The president did not mention Michael Jackson.
# SOURCES:

# QUESTION: {question}
# =========
# {summaries}
# =========
# FINAL ANSWER:"""
PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
