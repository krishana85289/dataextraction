from flask import Flask, request, jsonify
#from pdfminer.high_level import extract_text
from flask import Flask, request, jsonify
#from pdfminer.high_level import extract_text
import re
from flask import Flask, render_template
import csv
import defusedxml
#import json
from flask import Flask, send_file
import json
import re
import os
import time  # Added for timestamp
import uuid  # Added for unique identifier
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_extraction_chain_pydantic
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.llms import GooglePalm, OpenAI
from flask_cors import CORS
from langchain.llms import GooglePalm
from PIL import Image

from typing import Optional
from pydantic import BaseModel
from langchain.chains import create_extraction_chain_pydantic
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.llms import GooglePalm, OpenAI
#import pytesseract
import cv2
from langchain.llms import HuggingFaceHub
import pandas as pd
#from asposecells.api import Workbook, JsonLoadOptions
#pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Replace with the path to your

api_key = 'AIzaSyDifpKp5EwKvKT1ygGTPlG4J5UIGmp00vA' 

from doctr.io import DocumentFile
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(google_api_key=api_key,model="gemini-pro")

# llm = GooglePalm(google_api_key=api_key, temperature=0)  # GooglePalm model
## Invoice details 
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.llms import GooglePalm, OpenAI

## Invoice details 
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.llms import GooglePalm, OpenAI

## Invoice details 
invoice_info = ResponseSchema(
        name="Invoice number",
        description="In this include Invoice number, Invoice number can be in any form example : Inv no, Order no, etc",
    )
order_number_info = ResponseSchema(
    name="Order number",
    description="Details related to the order number, such as the unique identifier for an order. Example : order no, Order number",
)

Invoice_Date = ResponseSchema(
        name="Invoice Date",
        description="In this will show invoice date",
    )
Due_Date = ResponseSchema(
        name="Due Date",
        description="In this will show Due Date if you not found due  date so just set 'None',"
    )

order_date_info = ResponseSchema(
    name="Order date",
    description="Details related to the order date, such as the date when an order was placed.",
)
Company_Name = ResponseSchema(
    name="Company Name",
    description="This will show the name of the company",
)
Company_Address = ResponseSchema(
    name="Company Address",
    description="This will show the address of the company",
)
Company_Phone = ResponseSchema(
    name="Company Phone",
    description="This will show the phone number of the company",
)
Company_Email = ResponseSchema(
    name="Company Email",
    description="This will show the email address of the company",
)
shipping_information = ResponseSchema(
    name="Shipping Information",
    description = """Extracts shipping information from the invoice. Shipping information is included only when found in the text; otherwise, it is set to 'none'."

    example={
        "Recipient Name": "str",
        "Shipping Address": "str",
        "Shipping Method": "str",
        "Tracking Number": "str",
        "mobile number":"int"
    }
    """
)
company_information_schema = ResponseSchema(
    name="Company Information",
    description="Retrieve details related to Company's Information, which may include key details about a company or organization. Extract and provide values for the following fields: Name, Company Type, Registration Number, Industry, Contact Information, Website URL and other relevant details. Only provide values that are found within the Company Information section or any other source related to sellers Company.",
)

Billing_Address = ResponseSchema(
    name="Billing Address",
    description="This will show the billing address associated with the order",
)
Billing_Name = ResponseSchema(
    name="Billing Name",
    description="This will show the name associated with the billing information",
)
Billing_Phone = ResponseSchema(
    name="Billing Phone",
    description="This will show the phone number associated with the billing information",
)
Shipping_Address = ResponseSchema(
    name="Shipping Address",
    description="This will show the shipping address associated with the order",
)
Shipping_Name = ResponseSchema(
    name="Shipping Name",
    description="This will show the name associated with the shipping information",
)
Shipping_Phone = ResponseSchema(
    name="Shipping Phone",
    description="This will show the phone number associated with the shipping information",
)

Product_informations = ResponseSchema(
        name="Products information",
        description="Please provide details related to the Products list in a dictionary format. Extract and provide values for the following fields: 'Name,' 'SKU,' ''Price,' 'delivery date' ,'discount',' 'delivery Time' ,'tracking url','Quantity',and 'Subtotal' for each item. The JSON data should be structured as a dictionary with a key 'Items' that contains a list of dictionaries, each representing an item with these specified fields.",
)
Delivery_Fee = ResponseSchema(
    name="Delivery Fee",
    description="This will show the delivery fee for the order",
)
Tax_Rate = ResponseSchema(
    name="Tax Rate",
    description="This will show the tax rate applied to the order",
)
Order_Total = ResponseSchema(
    name="Order Total",
    description="This will show the total amount of the order, including all charges and fees",
)
Subtotal = ResponseSchema(
    name="Subtotal",
    description="This will show the subtotal amount of the order before any taxes or fees are applied",
)
Tax = ResponseSchema(
    name="Tax",
    description="This will show the tax amount applied to the order",
)
Currency_Symbol = ResponseSchema(
    name="Currency Symbol",
    description="This will show the symbol of the currency used for the order",
)


output_parser = StructuredOutputParser.from_response_schemas(
    [invoice_info,order_number_info,Invoice_Date,Due_Date,order_date_info, Company_Name,Company_Email,Company_Phone,Company_Address,Product_informations,Tax_Rate,Order_Total,Subtotal,Tax,Shipping_Address,Shipping_Name,Shipping_Phone,Billing_Phone,Billing_Name,Billing_Address,Currency_Symbol,Currency_Symbol]
)

response_format = output_parser.get_format_instructions()
print(response_format)

prompt = ChatPromptTemplate.from_template("Please provide a valid JSON format, JSON should follow the standard syntax and structure to be processed correctly. {processed_text} \n {response_format}"if response_format else "none")
app = Flask(__name__)

from doctr.models import ocr_predictor

model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
from doctr.io import DocumentFile


    
def replace_null_with_none(obj):
    if isinstance(obj, dict):
        return {key: replace_null_with_none(value) for key, value in obj.items() if value is not None}
    elif isinstance(obj, list):
        return [replace_null_with_none(element) for element in obj if element is not None]
    else:
        return obj

def check_invoice_keywords(text):
    keywords = ['invoice', 'order', 'date', 'price']

    # Check if any keyword is present in the text
    if any(keyword in text.lower() for keyword in keywords):
        return True
    else:
        return False


def create_document(file_path):
    # Check the file type based on the file extension
    if file_path.lower().endswith('.pdf'):
        # Create a DocumentFile from a PDF
        document = DocumentFile.from_pdf(file_path)
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # Create a DocumentFile from an image
        document = DocumentFile.from_images(file_path)
    else:
        raise ValueError("Unsupported file type. Supported types: PDF, JPEG, PNG, GIF")
    
    return document

CORS(app)
UPLOAD_FOLDER = r'UPLOAD_FOLDER'
JSON_FOLDER = r'JSON_FOLDER'
CSV_FOLDER = r'CSV_FOLDER'
IMAGE_UPLOAD=r'IMAGE_UPLOAD'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSON_FOLDER'] = JSON_FOLDER
app.config['CSV_FOLDER'] = CSV_FOLDER


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/v1/img-text', methods=['POST'])
def extract_text_from_img():
    files = request.files.getlist('image')
    if 'image' not in request.files:
        return jsonify({"error": "Key should be image"}), 400
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400
    aggregated_data = []
    user_ip = request.remote_addr
    try:
        for file in files:
            fname = file.filename
            
            name, extension = fname.rsplit(".", 1)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_identifier = str(uuid.uuid4().hex[:6])
            new_filename = f"{name}_{timestamp}_{unique_identifier}.{extension}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)

            file.save(file_path)

            document = create_document(file_path)
            result = model(document)
            #image = result.show(document)

            text = result.render()
            processed_text = " ".join(text.split("\n"))
            recipe = processed_text
            if not check_invoice_keywords(recipe):
                return jsonify({"error": "Invalid Invoice Document: Please add valid invioce document"}), 400
            formated_prompt = prompt.format(**{"processed_text": recipe, "response_format": output_parser.get_format_instructions()})
            response_palm = llm.invoke(formated_prompt)
            response_palm = response_palm.content

            cleaned_json = re.sub(r'\s{4,}', '', response_palm)
            #cleaned_json = re.sub(r'\$', '', response_palm)
            cleaned_json = re.sub(r',\s+', ', ', cleaned_json)
            cleaned_json = re.sub(r'\n', '', cleaned_json)
            cleaned_json = re.sub(r'\t', '', cleaned_json)
            cleaned_json = re.sub(r'<json>|<\/json>', '', cleaned_json)
            cleaned_json = re.sub(r'```json', '', cleaned_json)
            cleaned_json = re.sub(r'```', '', cleaned_json)
            cleaned_json = re.sub(r'\?', '', cleaned_json)

            data_dict = json.loads(cleaned_json)

            aggregated_data.append(data_dict)
            data=replace_null_with_none(aggregated_data)

        return jsonify({"result": data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8181, debug=True)