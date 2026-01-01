from typing import List
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
import os
import base64
from textwrap import dedent
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()


class Address(BaseModel):
    name: str = Field(description="the name of person and organization")
    address_line: str = Field(
        description="the local delivery information such as street, building number, PO box, or apartment portion of a postal address"
    )
    city: str = Field(description="the city portion of the address")
    state_province_code: str = Field(description="the code for address US states")
    postal_code: int = Field(description="the postal code portion of the address")


class Product(BaseModel):
    product_description: str = Field(
        description="the description of the product or service"
    )
    count: int = Field(description="number of units bought for the product")
    unit_item_price: float = Field(description="price per unit")
    product_total_price: float = Field(
        description="the total price, which is number of units * unit_price"
    )


class TotalBill(BaseModel):
    total: float = Field(description="the total amount before tax and delivery charges")
    discount_amount: float = Field(
        description="discount amount is total cost * discount %"
    )
    tax_amount: float = Field(
        description="tax amount is tax_percentage * (total - discount_amount). If discount_amount is 0, then its tax_percentage * total"
    )
    delivery_charges: float = Field(description="the cost of shipping products")
    final_total: float = Field(
        description="the total price or balance after removing tax, adding delivery and tax from total"
    )


class Invoice(BaseModel):
    invoice_number: str = Field(
        description="extraction of relevant information from invoice"
    )
    billing_address: Address = Field(
        description="where the bill for a product or service is sent so it can be paid by the recipient"
    )
    product: List[Product] = Field(description="the details of bill")
    total_bill: TotalBill = Field(
        description="the details of total amount, discounts and tax"
    )


class ImageLoaderBase64:

    def __init__(self, user_prompt: str, image_file_path: str):
        with open(image_file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        self.encoded_message_with_image = [
            {"type": "text", "text": f"{user_prompt}"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high",
                },
            },
        ]


system_prompt = """
    You are an assistant that extracts details from invoices.
"""

agent = Agent(
    model=OpenAIModel(
        model_name=os.getenv("OLLAMA_MODEL"), base_url="http://localhost:11434/v1"
    ),
    result_type=Invoice,
    system_prompt=dedent(system_prompt),
)

image_request_1 = ImageLoaderBase64(
    user_prompt="extract structure from the provided image",
    image_file_path=os.path.expanduser(
        "~/Documents/genai-training-pydanticai/data/vision_data/invoice.png"
    ),
)

response = agent.run_sync(image_request_1.encoded_message_with_image)
print(response.data)
